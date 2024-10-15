import numpy as np
import argparse
import nibabel as nib
from dipy.core.gradients import gradient_table

import TrackToLearn.algorithms.so3_helper as soh
import TrackToLearn.utils.torch_shorthands


def convert(x, grad_dirs):
    B_e3nn, invB_e3nn = soh.e3nn_sh_to_sf_matrix(
        TrackToLearn.utils.torch_shorthands.tt(grad_dirs),
        lmax=6,
        basis_type='symmetric'
        )

    y = np.matmul(x, invB_e3nn)

    return y

def add_parser_args(parser):
    parser.add_argument('dwi', help='input diffusionn weighted image')
    parser.add_argument('out', help='dwi output (image with e3nn basis sp harmonics)')
    parser.add_argument('bvals', help='b-value textfile')
    # parser.add_argument('bvecs', help='b-vector textfile')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_args(parser)
    args = parser.parse_args()

    # load input
    img_in = nib.load(args.dwi)
    data_in = img_in.get_fdata()

    # load bvals, bvecs
    bv = np.loadtxt(args.bvals)
    gt = gradient_table(bvals=bv[..., -1], bvecs=bv[..., :-1])

    import ipdb; ipdb.set_trace()

    # gt = gradient_table(bvals=args.bvals, bvecs=args.bvecs)
    dirs = gt.bvecs[[not val for val in gt.b0s_mask]]

    # import ipdb; ipdb.set_trace()

    # remove b0 from data and normalize by b0
    data_in_normalized = data_in[..., [not val for val in gt.b0s_mask]] / data_in[..., gt.b0s_mask]

    # transform basis
    data_out = convert(data_in_normalized, dirs)

    # save output
    img_out = nib.Nifti1Image(data_out,
                              img_in.affine,
                              img_in.header)
    nib.save(img_out, args.out)
