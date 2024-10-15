import numpy as np
from sys import argv
import nibabel as nib


def convert(x):
    y = x[..., :15]

    return y


if __name__ == '__main__':
    dwi_in = argv[1]
    dwi_out = argv[2]
    # dwi_in = '/fabi_project/data/ttl_anat_priors/raw_fabi_2024/fibercup/fodfs/fibercup_fodf.nii.gz'
    # dwi_out = '/fabi_project/data/ttl_anat_priors/raw_fabi_2024/fibercup/l_power_slope/fibercup_lps.nii.gz'

    # load input
    img_in = nib.load(dwi_in)
    data_in = img_in.get_fdata()

    # transform basis
    data_out = convert(data_in)

    # save output
    img_out = nib.Nifti1Image(data_out,
                              img_in.affine,
                              img_in.header)
    nib.save(img_out, dwi_out)
