# Release scope

This repository packages the core definitions used by the manuscript. Original model constructor and forward definitions, codec, convolution/adjoint and CG routines are retained. Imports are made package-relative, experiment-specific dynamic source loading and metadata/controller code are omitted, and the sampler exposes only fixed-kernel reconstruction. The training loss and node-storage adapter are small portable wrappers around the published formulas. Legacy configuration fields in `ModelConfig` are retained for compatibility but are not physical-parameter heads of the released root-residual predictor.

The package is checked with synthetic CPU fixtures. It has not rerun the 200-source comparison, replaced the archived experiment implementation or established cross-backend bit identity. Actual trained weights and the fitted codec are required for real model use. The image-prior model must be obtained separately under its own license. Never load untrusted pickle checkpoints.

The public package excludes server addresses, personal file paths, credentials, internal logs, private datasets, target arrays, controller/grant files, binaries and large checkpoints. Detailed operational provenance remains in the private research handoff rather than in the paper or repository.
