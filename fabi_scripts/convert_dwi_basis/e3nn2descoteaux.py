import numpy as np
from sys import argv
import nibabel as nib
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.reconst import dti
from dipy.direction import Sphere
from dipy.core.gradients import GradientTable

import TrackToLearn.algorithms.so3_helper as soh
import TrackToLearn.utils.torch_shorthands
from TrackToLearn.utils.rotation import get_ico_points


def derrive_anositropy_measure(x, v, measure='fractional_anisotropy'):
    gtab = GradientTable(v, b0_threshold=0.)
    tmodel = dti.TensorModel(gtab)
    tfit = tmodel.fit(x)
    mfun = getattr(dti, measure)

    return mfun(tfit.evals)


def convert(x):
    v = get_ico_points()
    B_e3nn, invB_e3nn = soh.e3nn_sh_to_sf_matrix(
        TrackToLearn.utils.torch_shorthands.tt(v),
        lmax=4,
        basis_type='symmetric'
        )
    B_desc, invB_desc = sh_to_sf_matrix(
        Sphere(xyz=v),
        sh_order=4,
        basis_type='descoteaux07',
        full_basis=False,
        legacy=False,
        return_inv=True,
        smooth=0
        )
    
    radii =  np.matmul(x, B_e3nn)

    y = np.matmul(
            radii,
        invB_desc
        )

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
