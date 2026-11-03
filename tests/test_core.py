import unittest
from unittest.mock import patch
import numpy as np
import torch
from skfdiff import FrozenRootPCACodec, SuperKernelPredictor
from skfdiff.base import trainable_parameter_count
from skfdiff.operators import ContinuousBlur, ConsistentBlur, nodes_to_field, untile
from skfdiff.losses import field_loss
from skfdiff import reconstruction as rec


torch.set_num_threads(2)


class CoreChecks(unittest.TestCase):
    def test_codec_and_uniform_fallback(self):
        codec=FrozenRootPCACodec(torch.zeros(36),torch.zeros(3,36),torch.ones(3),grid=2,kernel_size=3)
        out=codec(torch.zeros(1,3))
        self.assertEqual(tuple(out.shape),(1,2,2,3,3))
        self.assertTrue(torch.allclose(out.sum((-2,-1)),torch.ones(1,2,2)))
        self.assertTrue(torch.all(out>=0))

    def test_spectral_token_shape_and_flat_image(self):
        spectra=SuperKernelPredictor.spectral_features(torch.zeros(1,3,256,256))
        self.assertEqual(tuple(spectra.shape),(1,64,544))
        self.assertTrue(torch.equal(spectra,torch.zeros_like(spectra)))

    def test_model_parameter_count(self):
        codec=FrozenRootPCACodec(torch.ones(64*961),torch.zeros(256,64*961),torch.ones(256))
        model=SuperKernelPredictor(codec=codec)
        self.assertEqual(trainable_parameter_count(model),24775329)
        self.assertEqual(torch.count_nonzero(model.node_head[-1].weight).item(),0)
        self.assertTrue(all(p.requires_grad for p in model.base.parameters()))

    def test_delta_operator_storage_and_cg(self):
        nodes=torch.zeros(1,8,8,31,31);nodes[...,15,15]=1
        op=ContinuousBlur(nodes_to_field(nodes))
        x=torch.linspace(0,1,3*256*256).reshape(1,3,256,256)
        self.assertTrue(torch.allclose(op.forward(x),x,atol=2e-6))
        prior=torch.zeros_like(x);rho=.3
        actual,info=op.prox(x,prior,rho,iterations=12)
        self.assertTrue(torch.allclose(actual,x/(1+rho),atol=3e-6))
        self.assertLessEqual(info['cg_steps'],12)

    def test_spatially_varying_adjoint(self):
        gen=torch.Generator().manual_seed(17)
        nodes=torch.rand(1,1,2,2,31,31,generator=gen,dtype=torch.float64)
        nodes=nodes/nodes.sum((-2,-1),keepdim=True)
        field=untile(torch.nn.functional.pad(nodes,(1,0,1,0))).expand(-1,3,-1,-1).contiguous()
        op=ContinuousBlur(field)
        x=torch.randn(1,3,64,64,generator=gen,dtype=torch.float64)
        y=torch.randn(1,3,64,64,generator=gen,dtype=torch.float64)
        lhs=(op.forward(x)*y).sum();rhs=(x*op.adjoint(y)).sum()
        self.assertLess(abs(float(lhs-rhs)),1e-10)

    def test_loss_labels_and_blur_normalization(self):
        target=torch.ones(1,1,1,3,3)/9;pred=target.clone()
        x=torch.ones(1,3,4,4);clean=x*.5
        loss,parts=field_loss(pred,target,x,clean,lambda sharp,k:sharp*.5)
        self.assertEqual(loss.item(),0.)
        loss,_=field_loss(pred,target,x,clean,lambda sharp,k:sharp*.6)
        self.assertAlmostEqual(loss.item(),.04,places=6)

    def test_sampler_endpoint_and_count(self):
        calls={'model':0,'prox':0}
        class FakeOp:
            def __init__(self,*args):pass
            def prox(self,y,prior,rho,iterations):
                calls['prox']+=1
                return torch.zeros_like(y)+.25,{'cg_steps':0,'normal_equation_relative_residual':0.}
        def fake_prior(z,t):
            calls['model']+=1
            a=rec.alpha_bar()[int(t.item())]
            return (z/np.sqrt(a)-.4)/np.sqrt(1/a-1)
        nodes=torch.zeros(1,8,8,31,31);nodes[...,15,15]=1
        with patch.object(rec,'ContinuousBlur',FakeOp):
            result,trace,_=rec.reconstruct(fake_prior,torch.ones(1,3,256,256)*.5,nodes)
        self.assertTrue(torch.allclose(result,torch.full_like(result,.7),atol=2e-5))
        self.assertEqual(calls,{'model':100,'prox':100})
        self.assertEqual([trace[0]['t'],trace[-1]['t']],[200,0])


if __name__=='__main__':unittest.main()
