# Method and training settings

1. A five-scale convolutional encoder pools spatial features onto 64 tokens. Attention pooling predicts 256 whole-field PCA root coordinates.
2. The RGB-mean observation is split into 64 patches of 32×32 pixels. Each normalized log-power spectrum has 544 real-FFT coefficients. Spectral tokens are fused with spatial tokens and processed by two attention blocks.
3. A shared node head outputs 961 residual-root values per node. Its last linear layer starts at zero. The inherited spatial encoder remains trainable; the codec is frozen.
4. The summed root is mapped to `relu(root)^2` and normalized per node, with a uniform fallback for mass at most 1e-12.
5. Training minimizes mean nodewise PSF L1 plus normalized noiseless operator-response MSE. Ground-truth PSFs and sharp images are supervised targets only.
6. The full predictor is evaluated once at inference. No restored image, family label or ground-truth kernel is passed to it. A separate frozen image prior reconstructs under the fixed predicted field.

The reported predictor has 24,775,329 trainable parameters. Training adds 16,000 updates at effective batch 32, using fresh AdamW, inherited/new learning rates 1e-5/5e-5, betas (0.9,0.999), epsilon 1e-8, weight decay 1e-4, gradient norm cap 1, warmup 500 and cosine decay to 0.1. The EMA decay is 0.999. Development calibration selects the final eligible EMA at step 16,000. The inherited model was pretrained; this budget is additional exposure, not total training from scratch.

The root residual removes an explicit affine PCA-subspace restriction. It does not guarantee universal PSF representation, identifiable physical parameters, or a globally optimal blind inverse. The finite CG solve is unconstrained, and the stochastic reconstruction is not exact posterior sampling.
