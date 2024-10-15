import numpy as np
import argparse
import nibabel as nib
from dipy.core.gradients import gradient_table

import TrackToLearn.utils.torch_shorthands
from utils import get_ico_points

import TrackToLearn.algorithms.so3_helper as soh


def convert(x, subdiv):
    v = get_ico_points(subdiv)
    B_e3nn, invB_e3nn = soh.e3nn_sh_to_sf_matrix(
        TrackToLearn.utils.torch_shorthands.tt(v),
        lmax=6,
        basis_type='symmetric'
        )

    y = np.matmul(x, B_e3nn)

    return y

def add_parser_args(parser):
    parser.add_argument('in_sh', help='input sh image')
    parser.add_argument('out', help='pts output')
    parser.add_argument('subdiv', help='icosaeder subdivisions')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_args(parser)
    args = parser.parse_args()

    # load input
    img_in = nib.load(args.in_sh)
    data_in = img_in.get_fdata()

    # transform basis
    data_out = convert(data_in, args.subdiv)

    # save output
    img_out = nib.Nifti1Image(data_out,
                              img_in.affine,
                              img_in.header)
    nib.save(img_out, args.out)
