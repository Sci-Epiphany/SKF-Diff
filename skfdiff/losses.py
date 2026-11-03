"""Supervised PSF-shape and operator-response loss, for training only."""
import torch


def field_loss(predicted, target, sharp, clean_response, forward_operator):
    """Target PSFs and sharp references are labels, never blind inputs.

    forward_operator(sharp,predicted) returns the noiseless reblur with the
    predicted 8x8x31x31 nodes. Reductions match the reported objective.
    """
    l1=(predicted-target).abs().sum((-2,-1)).mean()
    response=forward_operator(sharp,predicted)
    denom=(sharp-clean_response).square().mean((1,2,3)).clamp_min(1e-4)
    relative=((response-clean_response).square().mean((1,2,3))/denom).mean()
    return l1+relative, {'psf_l1':l1,'operator_relative_mse':relative}
