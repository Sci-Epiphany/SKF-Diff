"""Fixed-field sampler; caller supplies the frozen unconditional noise U-Net.

This is a portable extraction, not a pretrained prior download utility.
"""
import numpy as np
import torch
from .operators import canonicalize, ContinuousBlur, nodes_to_field


def alpha_bar():
    return np.cumprod(1. - np.linspace(1e-4, .02, 1000))


@torch.no_grad()
def reconstruct(model, observation, nodes, seed=0):
    """One prescribed 100-level trajectory; output is the final denoiser prior.

    model(z,t) must return [1,C,256,256], C>=3. Only its first three channels
    are used as predicted noise. This function does not update the PSF.
    """
    if observation.shape != (1,3,256,256):
        raise ValueError('expected batch-one RGB 256x256 observation')
    a=alpha_bar(); schedule=np.linspace(200,0,100).round().astype(int).tolist()
    diagonal=(observation[:,:,::2,::2]-observation[:,:,1::2,::2]
              -observation[:,:,::2,1::2]+observation[:,:,1::2,1::2])/2
    sigma=max(float(diagonal.abs().median())/.67448975,1/255.)
    rng=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.randn(observation.shape,generator=rng).to(observation.device)
    state=np.sqrt(a[200])*(observation*2-1)+np.sqrt(max(
        1-a[200]-a[200]*(2*sigma)**2,0.))*noise
    field=canonicalize(nodes_to_field(nodes),32); op=ContinuousBlur(field,32)
    trace=[]
    for i,t in enumerate(schedule):
        eps=model(state,torch.full((1,),t,device=state.device,dtype=torch.long))[:,:3]
        prior=(state/np.sqrt(a[t])-np.sqrt(1/a[t]-1)*eps).clamp(-1,1)
        rho=4*(2*sigma)**2/((1-a[t])/a[t])
        corrected,diagnostic=op.prox(observation,(prior+1)/2,rho,iterations=12)
        z=corrected*2-1; residual=(state-np.sqrt(a[t])*z)/np.sqrt(1-a[t])
        if i+1<len(schedule):
            fresh=torch.randn(observation.shape,generator=rng).to(observation.device)
            state=np.sqrt(a[schedule[i+1]])*z+np.sqrt(1-a[schedule[i+1]])*(
                np.sqrt(.9)*residual+np.sqrt(.1)*fresh)
        else:
            state=prior
        if not torch.isfinite(state).all():
            raise FloatingPointError('nonfinite state')
        trace.append(dict(iteration=i,t=t,**diagnostic))
    return (state+1)/2, trace, sigma


@torch.no_grad()
def reconstruct_mean(model, observation, nodes):
    """Fixed seeds 0 and 3; FP64 mean -> FP32, no quality-based selection."""
    images=[]; traces=[]
    for seed in (0,3):
        image,trace,_=reconstruct(model,observation,nodes,seed)
        images.append(image.cpu().numpy()); traces.append(trace)
    mean=np.stack(images).astype(np.float64).mean(0).astype(np.float32)
    return mean, traces
