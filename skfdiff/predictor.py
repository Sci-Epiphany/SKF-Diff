"""Spatial--spectral whole-field kernel predictor."""
from dataclasses import asdict
import torch
from torch import nn
from . import base as v61_models
ModelConfig=v61_models.ModelConfig
FrozenRootPCACodec=v61_models.FrozenRootPCACodec

def _count(module):
    return sum(parameter.numel() for parameter in module.parameters()
               if parameter.requires_grad)


class SuperKernelPredictor(nn.Module):
    """Warm-start-compatible PCA base plus unrestricted root-space residual.

    The sole model input is y[B,3,256,256]. All base network parameters remain
    trainable; the codec stays frozen. Zero initialization of the last residual
    layer makes the initial prediction exactly the base prediction. It also
    makes earlier residual-branch gradients zero on the first backward pass,
    which is expected and is not a claim of equal effective gradient capacity.
    """
    def __init__(self, codec_path=None, config=ModelConfig(), codec=None):
        super().__init__()
        if asdict(config) != asdict(ModelConfig()):
            raise ValueError("v86 uses the unchanged default v61 base architecture")
        self.config = config
        self.base = v61_models.RootPCAKernelPredictor(
            codec_path=codec_path, config=config, codec=codec)
        if any(value.dtype != torch.float32 for value in
               (self.base.codec.mean, self.base.codec.basis, self.base.codec.scale)):
            raise ValueError("v86 requires FP32 codec buffers")
        for parameter in self.base.parameters():
            parameter.requires_grad_(True)
        self.spectral = nn.Sequential(
            nn.LayerNorm(544), nn.Linear(544, 192), nn.GELU(), nn.Linear(192, 384))
        self.fusion = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 384), nn.GELU())
        self.node_blocks = nn.Sequential(
            v61_models._TokenBlock(384, 8, 4), v61_models._TokenBlock(384, 8, 4))
        self.node_head = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 512), nn.GELU(), nn.Linear(512, 961))
        nn.init.zeros_(self.node_head[-1].weight)
        nn.init.zeros_(self.node_head[-1].bias)

    @staticmethod
    def spectral_features(y):
        """Row-major 8x8 non-overlapping 32x32 RGB-mean patch spectra."""
        v61_models._check_y(y)
        with torch.autocast(device_type=y.device.type, enabled=False):
            gray = y.float().mean(dim=1)
            patches = gray.reshape(y.shape[0], 8, 32, 8, 32).permute(
                0, 1, 3, 2, 4).reshape(y.shape[0], 64, 32, 32)
            spectrum = torch.fft.rfft2(patches, dim=(-2, -1), norm="backward")
            power = spectrum.real.square() + spectrum.imag.square()
            mean_power = power.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            return torch.log1p(power / mean_power).flatten(start_dim=-2)

    def forward(self, y):
        evidence = self.base.encoder(y)
        with torch.autocast(device_type=y.device.type, enabled=False):
            coords = self.base.coord_head(self.base.decision(evidence["pooled"].float()))
            base_root = self.base.codec.roots(coords)
            spectral_tokens = self.spectral(self.spectral_features(y))
            tokens = self.fusion(torch.cat(
                (evidence["spatial_tokens"].float(), spectral_tokens), dim=-1))
            residual_root = self.node_head(self.node_blocks(tokens)).reshape(
                y.shape[0], 8, 8, 31, 31)
            positive = (base_root + residual_root).clamp_min(0).square()
            mass = positive.sum(dim=(-2, -1), keepdim=True)
            uniform = torch.full_like(positive, 1. / (31 ** 2))
            kernel = torch.where(mass > 1e-12,
                                 positive / mass.clamp_min(1e-12), uniform)
            return {"kernel": kernel, "coords": coords, "residual_root": residual_root}

