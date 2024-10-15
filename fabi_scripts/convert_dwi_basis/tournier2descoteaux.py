import numpy as np
from sys import argv
import nibabel as nib
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.reconst import dti
from dipy.direction import Sphere
from dipy.core.gradients import GradientTable

import TrackToLearn.algorithms.so3_helper as soh
from TrackToLearn.utils.rotation import get_ico_points
import TrackToLearn.utils.torch_shorthands


def convert(x, norm_scale=False, lmax=6):
    v = get_ico_points(subdiv=4)

    B_desc, invB_desc = sh_to_sf_matrix(
        Sphere(xyz=v),
        sh_order=lmax,
        basis_type='descoteaux07',
        full_basis=False,
        legacy=False,
        return_inv=True,
        smooth=0
    )

    B_tour, invB_tour = sh_to_sf_matrix(
        Sphere(xyz=v),
        sh_order=lmax,
        basis_type='tournier07',
        full_basis=False,
        legacy=False,
        return_inv=True,
        smooth=0
        )
    
    radii = np.matmul(x, B_tour)

    if norm_scale:
        rmin = radii.min(axis=-1, keepdims=True)
        rmax = radii.max(axis=-1, keepdims=True)
        radii = np.nan_to_num(((radii - rmin) / (2 * (rmax - rmin))))

    y = np.matmul(
            radii,
        invB_desc
        )

    return y


if __name__ == '__main__':
    lmax = 8

    dwi_in = argv[1]
    dwi_out = argv[2]
    # dwi_in = '/fabi_project/data/ttl_anat_priors/raw_fabi_2024/fibercup/fodfs/fibercup_fodf.nii.gz'
    # dwi_out = '/fabi_project/data/ttl_anat_priors/raw_fabi_2024/fibercup/l_power_slope/fibercup_lps.nii.gz'

    if len(argv) == 4 and argv[3] == 'norm':
        print('NORMALIZED')
        normalize = True
    else:
        normalize = False

    # load input
    img_in = nib.load(dwi_in)
    data_in = img_in.get_fdata()

    # transform basis
    data_out = convert(data_in, normalize, lmax=lmax)

    # save output
    img_out = nib.Nifti1Image(data_out,
                              img_in.affine,
                              img_in.header)
    nib.save(img_out, dwi_out)
