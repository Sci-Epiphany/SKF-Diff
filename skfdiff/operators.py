"""Whole-image bilinear PSF operator, exact adjoint and finite CG solve."""
import torch
import torch.nn.functional as F

def tiles(x, size):
    b,c,h,w=x.shape
    return x.reshape(b,c,h//size,size,w//size,size).permute(0,1,2,4,3,5)


def untile(x):
    b,c,gh,gw,h,w=x.shape
    return x.permute(0,1,2,4,3,5).reshape(b,c,gh*h,gw*w)


class UABCBlockBlur:
    def __init__(self,field,tile=32):
        self.tile=tile;self.batch,self.channels,self.height,self.width=field.shape
        self.gh=self.height//tile;self.gw=self.width//tile
        k=tiles(field,tile)[...,1:,1:]
        mass=k.sum((-2,-1),keepdim=True)
        if (k<0).any() or (mass<=0).any():raise ValueError('PSF must be nonnegative with positive mass')
        k=k/mass
        self.radius=(tile-1)//2;self.patch=tile+2*self.radius
        self.fft_size=1<<(self.patch-1).bit_length()
        self.h=torch.fft.fft2(k,s=(self.fft_size,self.fft_size))

    def forward(self,image):
        r=self.radius;p=self.patch
        padded=F.pad(image,(r,r,r,r),mode='constant',value=0.)
        patches=F.unfold(padded,kernel_size=p,stride=self.tile)
        patches=patches.reshape(self.batch,self.channels,p,p,self.gh,self.gw).permute(0,1,4,5,2,3)
        convolution=torch.fft.ifft2(torch.fft.fft2(patches,s=(self.fft_size,self.fft_size))*self.h).real
        start=2*r
        return untile(convolution[...,start:start+self.tile,start:start+self.tile])

    def adjoint(self,image):
        r=self.radius;p=self.patch;n=self.fft_size
        patches=F.pad(tiles(image,self.tile),(2*r,n-self.tile-2*r,2*r,n-self.tile-2*r))
        patches=torch.fft.ifft2(torch.fft.fft2(patches)*self.h.conj()).real[...,:p,:p]
        columns=patches.permute(0,1,4,5,2,3).reshape(self.batch,self.channels*p*p,self.gh*self.gw)
        padded=F.fold(columns,output_size=(self.height+2*r,self.width+2*r),kernel_size=p,stride=self.tile)
        return padded[:,:,r:r+self.height,r:r+self.width]

def canonicalize(field,tile):
    if not torch.isfinite(field).all():raise ValueError('Nonfinite PSF')
    p=tiles(field,tile)[...,1:,1:]
    if (p<0).any():raise ValueError('Negative PSF')
    mass=p.sum((-2,-1),keepdim=True)
    if (mass<=1e-12).any():raise ValueError('Empty effective PSF')
    return untile(F.pad(p/mass,(1,0,1,0)))


def gradient_normal(x):
    """D^T D on the whole image, including every internal block boundary."""
    dx=x[:,:,:,1:]-x[:,:,:,:-1]
    dy=x[:,:,1:,:]-x[:,:,:-1,:]
    return F.pad(dx,(1,0))-F.pad(dx,(0,1))+F.pad(dy,(0,0,1,0))-F.pad(dy,(0,0,0,1))


class ConsistentBlur(UABCBlockBlur):
    def __init__(self,field,tile):
        super().__init__(canonicalize(field,tile),tile=tile)

    def prox(self,measurement,prior,weight,iterations=12,gradient_ratio=0.):
        """Global CG solve; optional gradient prior acts inside inference, not after it.

        min ||Ax-y||^2 + rho ||x-z||^2 + rho*g ||D(x-z)||^2.
        The gradient is relative to the full-image denoiser prior, not a command
        to remove genuine edges. No special seam masking or block blending.
        """
        if weight<=0 or gradient_ratio<0:raise ValueError('Invalid regularization')
        def regularize(x):
            return weight*(x+gradient_ratio*gradient_normal(x)) if gradient_ratio else weight*x
        def system(x):return self.adjoint(self.forward(x))+regularize(x)
        rhs=self.adjoint(measurement)+regularize(prior)
        x=prior.clone();r=rhs-system(x);p=r.clone();dims=tuple(range(1,x.ndim))
        rs=r.square().sum(dims,keepdim=True);initial=rs.clone();steps=0
        for _ in range(iterations):
            if (rs<=initial*1e-8+1e-20).all():break
            ap=system(p);alpha=rs/(p*ap).sum(dims,keepdim=True).clamp_min(1e-20)
            x=x+alpha*p;r=r-alpha*ap;new=r.square().sum(dims,keepdim=True)
            p=r+(new/rs.clamp_min(1e-20))*p;rs=new;steps+=1
        residual=rhs-system(x)
        relative=float(residual.norm()/rhs.norm().clamp_min(1e-12))
        return x,{'cg_steps':steps,'normal_equation_relative_residual':relative}

class ContinuousBlur:
    def __init__(self,field,tile=32):
        if tile!=32:raise ValueError('This prototype fixes support/grid for every category')
        self.field=field;self.tile=tile
        p=tiles(field,tile);n=p.shape[2];idx=torch.arange(n,device=field.device)
        u=(torch.arange(field.shape[-1],device=field.device)%tile)/float(tile)
        v=(torch.arange(field.shape[-2],device=field.device)%tile)/float(tile)
        self.ops=[];self.weights=[]
        for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:
            shifted=p.index_select(2,(idx+dy).clamp_max(n-1)).index_select(3,(idx+dx).clamp_max(n-1))
            self.ops.append(ConsistentBlur(untile(shifted),tile))
            self.weights.append(((v if dy else 1-v)[:,None]*(u if dx else 1-u)[None,:])[None,None])

    def forward(self,x):return sum(w*op.forward(x) for w,op in zip(self.weights,self.ops))

    def adjoint(self,y):return sum(op.adjoint(w*y) for w,op in zip(self.weights,self.ops))

    # Same global CG system as the verified block operator, with this A/A^T.
    prox=ConsistentBlur.prox

def nodes_to_field(nodes):
    """[B,8,8,31,31] achromatic nodes -> [B,3,256,256] storage field.

    Padding is storage only. ContinuousBlur supplies actual image-neighbor
    context across the 32-pixel output blocks.
    """
    if nodes.ndim != 5 or tuple(nodes.shape[1:]) != (8,8,31,31):
        raise ValueError('expected [B,8,8,31,31] nodes')
    return untile(F.pad(nodes[:,None],(1,0,1,0))).expand(-1,3,-1,-1).contiguous()
