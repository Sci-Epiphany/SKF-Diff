"""Synthetic CPU operator demonstration, not a learned-restoration result."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from skfdiff.operators import ContinuousBlur, nodes_to_field


def main():
    torch.set_num_threads(2)
    nodes=torch.zeros(1,8,8,31,31)
    nodes[...,15,15]=1.
    image=torch.linspace(0,1,3*256*256).reshape(1,3,256,256)
    op=ContinuousBlur(nodes_to_field(nodes))
    y=op.forward(image)
    print('Synthetic delta-PSF identity maximum error:',float((y-image).abs().max()))
    print('No trained predictor, diffusion prior, real image or GPU was used.')


if __name__=='__main__':main()
