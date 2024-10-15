import numpy as np
from sys import argv
import nibabel as nib
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.reconst import dti
from dipy.direction import Sphere
from dipy.core.gradients import GradientTable

import TrackToLearn.algorithms.so3_helper as soh
import TrackToLearn.utils.rotation
import TrackToLearn.utils.torch_shorthands


def derrive_anositropy_measure(x, v, measure='fractional_anisotropy'):
    gtab = GradientTable(v, b0_threshold=0.)
    tmodel = dti.TensorModel(gtab)
    tfit = tmodel.fit(x)
    mfun = getattr(dti, measure)

    return mfun(tfit.evals)


def convert(x, norm_scale=False):
    v = TrackToLearn.utils.rotation.get_ico_points()
    B_e3nn, invB_e3nn = soh.e3nn_sh_to_sf_matrix(
        TrackToLearn.utils.torch_shorthands.tt(v),
        lmax=6,
        basis_type='symmetric'
        )
    B_desc, invB_desc = sh_to_sf_matrix(
        Sphere(xyz=v),
        sh_order=6,
        basis_type='descoteaux07',
        full_basis=False,
        legacy=False,
        return_inv=True,
        smooth=0
        )
    
    radii =  np.matmul(x, B_desc)

    import ipdb; ipdb.set_trace()

    if norm_scale:
        rmin = radii.min(axis=-1, keepdims=True)
        rmax = radii.max(axis=-1, keepdims=True)
        radii = np.nan_to_num(((radii - rmin) / (2 * (rmax - rmin))))

    y = np.matmul(
            radii,
        invB_e3nn
        )

    return y


if __name__ == '__main__':
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
    data_out = convert(data_in, normalize)

    # save output
    img_out = nib.Nifti1Image(data_out,
                              img_in.affine,
                              img_in.header)
    nib.save(img_out, dwi_out)
