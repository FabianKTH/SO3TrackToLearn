import torch
import numpy as np
from typing import List
from itertools import zip_longest
# TODO from functools import cache
# import vtk
# from vtk.numpy_interface import dataset_adapter as dsa

from e3nn import o3
from dipy.reconst.shm import smooth_pinv

from TrackToLearn.utils.torch_shorthands import tt, tnp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def e3nn_sh_to_sf_matrix(points, lmax, basis_type='full'):
    if basis_type == 'full':
        Y = [o3.spherical_harmonics(order, points, normalize=True, normalization='norm').T  # NOTE: was 'integral'
             for order in range(lmax + 1)]
    elif basis_type == 'symmetric':
        Y = [o3.spherical_harmonics(order, points, normalize=True, normalization='norm').T
             for order in range(0, lmax + 1, 2)]
    elif basis_type == 'asymmetric':
        Y = [o3.spherical_harmonics(order, points, normalize=True, normalization='norm').T
             for order in range(1, lmax, 2)]
    else:
        raise ValueError(f'basis type {basis_type} unknown!')

    B_ = tnp(torch.cat(Y))
    invB_ = smooth_pinv(B_, L=np.zeros(B_.shape[1]))

    return B_, invB_
