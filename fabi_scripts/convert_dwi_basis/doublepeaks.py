import numpy as np
import argparse
import nibabel as nib


def mirror_peaks(x):
    x = x.reshape(x.shape[:-1] + (-1, 3))  # unfold last dimension
    x_m = -x

    x_both = np.zeros(x.shape[:-2] + (2 * x.shape[-2], 3))

    # combine by stacking interleaved
    x_both[..., ::2, :] = x
    x_both[..., 1::2, :] = x_m

    # fold last dimension again
    x_both = x_both.reshape(x.shape[:-2] + (-1, ))

    return x_both


def add_parser_args(parser):
    parser.add_argument('in_peaks', help='input peaks')
    parser.add_argument('out', help='peaks output')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_args(parser)
    args = parser.parse_args()

    # load input
    img_in = nib.load(args.in_peaks)
    data_in = img_in.get_fdata()

    # miror all peaks to the other side of the q ball
    data_out = mirror_peaks(data_in)

    # save output
    img_out = nib.Nifti1Image(data_out,
                              img_in.affine,
                              img_in.header)
    nib.save(img_out, args.out)
