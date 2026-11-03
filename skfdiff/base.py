"""Observation encoder and frozen square-root PCA codec."""
from dataclasses import dataclass, asdict
import io, hashlib, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 256
    channels: tuple = (64, 128, 256, 384, 512)
    depths: tuple = (2, 2, 3, 3, 3)
    expansion: int = 4
    fusion_channels: int = 96
    token_width: int = 384
    token_blocks: int = 2
    attention_heads: int = 8
    head_width: int = 512
    token_grid: int = 8
    pca_rank: int = 256
    parameterization: str = "flexible37"
    trajectory_bound: float = 8.0

    @property
    def theta_dim(self):
        return 37 if self.parameterization == "flexible37" else 22

    @property
    def raw_theta_dim(self):
        return 37 if self.parameterization == "flexible37" else 24

    def validate(self):
        if self.image_size != 256 or self.token_grid != 8:
            raise ValueError("v61 frozen geometry is RGB256 with spatial8x8 tokens")
        if len(self.channels) != 5 or len(self.depths) != 5:
            raise ValueError("exactly five encoder scales are required")
        dimensions = tuple(self.channels) + tuple(self.depths) + (
            self.expansion, self.fusion_channels, self.token_width,
            self.token_blocks, self.attention_heads, self.head_width, self.pca_rank)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if self.token_width % self.attention_heads:
            raise ValueError("token width must be divisible by attention heads")
        if self.parameterization not in ("native22", "flexible37"):
            raise ValueError("unsupported physical parameterization")
        if not math.isfinite(float(self.trajectory_bound)) or self.trajectory_bound <= 0:
            raise ValueError("trajectory_bound must be finite and strictly positive")


def _groups(width):
    return next(g for g in (8, 4, 2, 1) if width % g == 0)


def _finite_floating(value, name):
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise ValueError(name + " must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")


def _check_y(y):
    if not isinstance(y, torch.Tensor) or y.ndim != 4 or tuple(y.shape[1:]) != (3, 256, 256):
        raise ValueError("blind model input must have shape[B,3,256,256]")
    if y.shape[0] < 1:
        raise ValueError("empty image batches are unsupported")
    _finite_floating(y, "y")


class _SpatialBlock(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.GroupNorm(_groups(width), width)
        self.expand = nn.Conv2d(width, width * expansion, 1)
        self.contract = nn.Conv2d(width * expansion, width, 1)
        self.layer_scale = nn.Parameter(torch.full((width, 1, 1), .01))

    def forward(self, value):
        update = self.contract(F.gelu(self.expand(self.norm(self.depthwise(value)))))
        return value + self.layer_scale * update


class _TokenBlock(nn.Module):
    def __init__(self, width, heads, expansion):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * expansion), nn.GELU(),
                                 nn.Linear(width * expansion, width))

    def forward(self, value):
        norm = self.norm1(value)
        value = value + self.attention(norm, norm, norm, need_weights=False)[0]
        return value + self.mlp(self.norm2(value))


class KernelEvidenceEncoder(nn.Module):
    """Trainable raw-y multi-scale encoder; spatial positions survive pooling.

    forward returns pyramid scales128/64/32/16/8, a fused8x8 map, 64 position-
    aware spatial tokens, and an attention-pooled global vector. No tensor is
    detached. No hidden RNG is consumed in forward (all dropout is zero).
    """
    def __init__(self, config=ModelConfig()):
        super().__init__()
        config.validate()
        self.config = config
        widths = config.channels
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 5, stride=2, padding=2),
            nn.GroupNorm(_groups(widths[0]), widths[0]), nn.GELU())
        self.stages = nn.ModuleList(nn.Sequential(*[
            _SpatialBlock(width, config.expansion) for _ in range(depth)])
            for width, depth in zip(widths, config.depths))
        self.downsamples = nn.ModuleList(nn.Sequential(
            nn.GroupNorm(_groups(widths[i]), widths[i]),
            nn.Conv2d(widths[i], widths[i+1], 3, stride=2, padding=1)) for i in range(4))
        self.laterals = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, config.fusion_channels, 1), nn.GELU()) for width in widths)
        self.fusion = nn.Sequential(nn.Conv2d(5 * config.fusion_channels, config.token_width, 1),
            nn.GroupNorm(_groups(config.token_width), config.token_width), nn.GELU())
        axis = torch.linspace(-1., 1., config.token_grid)
        row = axis[:, None].expand(config.token_grid, config.token_grid)
        col = axis[None, :].expand(config.token_grid, config.token_grid)
        self.register_buffer("spatial_coordinates", torch.stack((row, col), dim=-1).reshape(1, -1, 2))
        self.coordinate_projection = nn.Linear(2, config.token_width, bias=False)
        self.position = nn.Parameter(torch.randn(1, config.token_grid ** 2, config.token_width) * .02)
        self.token_blocks = nn.Sequential(*[_TokenBlock(config.token_width,
            config.attention_heads, config.expansion) for _ in range(config.token_blocks)])
        self.token_norm = nn.LayerNorm(config.token_width)
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.token_width) * .02)
        self.pool_attention = nn.MultiheadAttention(config.token_width,
            config.attention_heads, dropout=0., batch_first=True)
        self.pool_norm = nn.LayerNorm(config.token_width)

    def forward(self, y):
        _check_y(y)
        value = self.stem(y)
        pyramid = []
        for index, stage in enumerate(self.stages):
            if index:
                value = self.downsamples[index-1](value)
            value = stage(value)
            pyramid.append(value)
        fused = self.fusion(torch.cat([lateral(F.adaptive_avg_pool2d(feature,
            (self.config.token_grid, self.config.token_grid)))
            for lateral, feature in zip(self.laterals, pyramid)], dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        coordinates = self.spatial_coordinates.to(dtype=tokens.dtype)
        tokens = self.token_norm(self.token_blocks(tokens + self.position +
            .1 * self.coordinate_projection(coordinates)))
        query = self.pool_query.expand(y.shape[0], -1, -1)
        pooled = self.pool_norm(self.pool_attention(query, tokens, tokens,
                                                   need_weights=False)[0][:, 0])
        return {"spatial_tokens": tokens, "pooled": pooled,
                "fused_map": fused, "pyramid": tuple(pyramid)}


def _decision_trunk(config):
    return nn.Sequential(nn.LayerNorm(config.token_width),
        nn.Linear(config.token_width, config.head_width), nn.GELU(),
        nn.Linear(config.head_width, config.head_width), nn.GELU())


class FrozenRootPCACodec(nn.Module):
    """Frozen absolute root-PCA decode; no trainable/legacy encoder dependency."""
    def __init__(self, mean, basis, scale, grid=8, kernel_size=31):
        super().__init__()
        dim = grid * grid * kernel_size * kernel_size
        if mean.shape != (dim,) or basis.ndim != 2 or basis.shape[1] != dim:
            raise ValueError("invalid root-PCA mean/basis shape")
        if scale.shape != (basis.shape[0],):
            raise ValueError("invalid root-PCA scale shape")
        for name, value in (("mean", mean), ("basis", basis), ("scale", scale)):
            _finite_floating(value, name)
            self.register_buffer(name, value.detach().clone())
        if not bool((scale > 0).all()):
            raise ValueError("root-PCA scales must be strictly positive")
        self.grid, self.kernel_size, self.rank = grid, kernel_size, basis.shape[0]

    @classmethod
    def from_npz(cls, path):
        path = Path(path)
        payload = path.read_bytes()
        codec = cls.from_npz_bytes(payload)
        codec.source_path = str(path.resolve())
        return codec

    @classmethod
    def from_npz_bytes(cls, payload):
        """Decode/hash the SAME byte receipt, avoiding a second filesystem read."""
        if not isinstance(payload, bytes):
            raise TypeError("codec payload must be immutable bytes")
        with np.load(io.BytesIO(payload), allow_pickle=False) as bank:
            for name in ("mean", "basis", "scale"):
                if name not in bank:
                    raise ValueError("codec missing " + name)
            tensors = [torch.from_numpy(np.array(bank[name], dtype=np.float32, copy=True))
                       for name in ("mean", "basis", "scale")]
        codec = cls(*tensors)
        codec.source_sha256 = hashlib.sha256(payload).hexdigest()
        return codec

    def roots(self, coords):
        if coords.ndim != 2 or coords.shape[1] != self.rank:
            raise ValueError("coords must have shape[B,codec_rank]")
        _finite_floating(coords, "PCA coords")
        dtype = torch.float64 if any(value.dtype == torch.float64 for value in
            (coords, self.mean, self.basis, self.scale)) else torch.float32
        with torch.autocast(device_type=coords.device.type, enabled=False):
            roots = self.mean.to(dtype=dtype) + (coords.to(dtype=dtype) *
                self.scale.to(dtype=dtype)) @ self.basis.to(dtype=dtype)
            return roots.reshape(coords.shape[0], self.grid, self.grid, self.kernel_size, self.kernel_size)

    def forward(self, coords):
        with torch.autocast(device_type=coords.device.type, enabled=False):
            positive = self.roots(coords).clamp_min(0).square()
            mass = positive.sum((-2, -1), keepdim=True)
            uniform = torch.full_like(positive, 1. / self.kernel_size ** 2)
            return torch.where(mass > 1e-12, positive / mass.clamp_min(1e-12), uniform)


class RootPCAKernelPredictor(nn.Module):
    """Strong raw-y control using the SAME encoder and frozen PCA geometry.

    Supply exactly one of codec_path (production) or codec (in-memory toy/tests).
    Instantiate both predictors with the same RNG seed, or explicitly clone
    encoder.state_dict(), to match encoder initialization. Models never share
    trainable parameter storage implicitly.
    """
    def __init__(self, codec_path=None, config=ModelConfig(), codec=None):
        super().__init__()
        config.validate()
        if (codec_path is None) == (codec is None):
            raise ValueError("supply exactly one codec_path or in-memory codec")
        self.config = config
        self.encoder = KernelEvidenceEncoder(config)
        self.decision = _decision_trunk(config)
        self.coord_head = nn.Linear(config.head_width, config.pca_rank)
        if codec_path is not None:
            self.codec = FrozenRootPCACodec.from_npz(codec_path)
            self.codec_path = self.codec.source_path
            self.codec_sha256 = self.codec.source_sha256
        else:
            if not isinstance(codec, FrozenRootPCACodec):
                raise TypeError("in-memory codec must be FrozenRootPCACodec")
            self.codec = codec
            self.codec_path = self.codec_sha256 = None
        if self.codec.rank != config.pca_rank or self.codec.grid != 8 or self.codec.kernel_size != 31:
            raise ValueError("control codec must match PCA rank and physical8x8x31x31 geometry")
        if any(p.requires_grad for p in self.codec.parameters()):
            raise ValueError("control codec cannot be trainable")

    def forward(self, y):
        evidence = self.encoder(y)
        with torch.autocast(device_type=y.device.type, enabled=False):
            coords = self.coord_head(self.decision(evidence["pooled"].float()))
            return {"coords": coords, "kernel": self.codec(coords)}

    def model_metadata(self):
        return {"config": asdict(self.config), "codec_path": self.codec_path,
                "codec_sha256": self.codec_sha256,
                "trainable_parameters": trainable_parameter_count(self),
                "encoder_trainable_parameters": trainable_parameter_count(self.encoder),
                "blind_inputs": ["y"], "posterior_implemented": False,
                "precision_policy": "encoder may autocast;decision/coordinate heads and complete root-PCA codec FP32",
                "legacy_encoder_calls": 0, "image_prior_NFE": 0,
                "physical_renderer_calls_inside_model": 0}


def trainable_parameter_count(model):
    return sum(value.numel() for value in model.parameters() if value.requires_grad)
