from os.path import join
import numpy as np
import shutil
from e3nn import o3
import h5py
from TrackToLearn.environments.interpolation import interpolate_volume_at_coordinates
import torch
from TrackToLearn.utils.torch_shorthands import tt, tnp
import nibabel as nib


def get_ico_points(subdiv=1):
    fname = f'/fabi_project/sphere/ico{subdiv}.npy'
    numpy_array_of_points = np.load(fname)

    return numpy_array_of_points


def rotate_data(angles, data, is_binary, in_irreps=None, return_trafo=False):
    rmat = o3.angles_to_matrix(angles[0],
                               angles[1],
                               torch.tensor([0.], device=torch.device('cuda')))[0]

    # make grid
    dimx, dimy, dimz, irr_dim = data.shape
    x, y, z = np.meshgrid(
        np.linspace(0, dimx-1, dimx),
        np.linspace(0, dimy-1, dimy),
        np.linspace(0, dimz-1, dimz)
    , indexing='ij')
    idx = np.stack([x, y, z], axis=-1)

    # data = data.reshape(-1, irr_dim)
    idx = idx.reshape(-1, 3)

    # center idx
    idx -= np.array([dimx/2, dimy/2, dimz/2])[None]

    # rotate volume coordinates
    idx_r = np.einsum('xi,ij->xj', idx, tnp(rmat)) # tnp(rmat.T))

    # move to old center
    idx_r += np.array([dimx/2, dimy/2, dimz/2])[None]

    # sample volume
    spline_order = 0 if is_binary else 1
    data_r = interpolate_volume_at_coordinates(data, idx_r, order=spline_order)

    # rotate also the sph/vectors
    if in_irreps is not None:
        assert o3.Irreps(in_irreps).dim == irr_dim

        # symmetrical spharm with lmax=6
        irreps = o3.Irreps(in_irreps)
        D_ = irreps.D_from_matrix(rmat.cpu())
        data_r = data_r @ tnp(D_.T)
        # data_r = data_r @ tnp(D_)
    elif irr_dim == 15:  # 3x5 peaks
        # vectors
        irreps = o3.Irreps("5x1o")
        D_ = irreps.D_from_matrix(rmat.cpu())
        data_r = data_r @ tnp(D_.T)
        # data_r = data_r @ tnp(D_)

    data_r = data_r.reshape(dimx, dimy, dimz, irr_dim)

    if return_trafo:
        return data_r, np.array([dimx/2, dimy/2, dimz/2]), rmat.cpu().numpy()
    else:
        return data_r


def pad_data(data):
    dimx, dimy, dimz, irr_dim = data.shape

    dimmax = np.max([dimx, dimy, dimz])

    p_beforex, p_beforey, p_beforez = ((dimmax - dimx)//2,
                                       (dimmax - dimy)//2,
                                       (dimmax - dimz)//2)
    p_afterx, p_aftery, p_afterz = (dimmax - (p_beforex + dimx),
                                    dimmax - (p_beforey + dimy),
                                    dimmax - (p_beforez + dimz))
    pad_width = ((p_beforex, p_afterx),
                 (p_beforey, p_aftery),
                 (p_beforez, p_afterz),
                 (0, 0))

    return np.pad(data, pad_width, mode='constant')  # const 0 by default


def rotate_hdf5(path_h5, orig_id, file_mask_in, path_mask_out, split_id='validation', in_irreps="1x0e+1x2e+1x4e+1x6e"):
    # get all xyz points
    xyz = get_ico_points(subdiv=1)
    alphas, betas = o3.xyz_to_angles(tt(xyz))

    # --

    # just for testing TODO remove
    alphas, betas = torch.zeros_like(alphas), torch.zeros_like(betas)
    alphas = torch.arange(-torch.pi/4, torch.pi/4, torch.pi/24).cuda()

    # --

    rot_params = dict()

    for rot_idx, (xyz_, alpha, beta) in enumerate(zip(xyz, alphas, betas)):
        with h5py.File(path_h5, 'r+') as file_h5:
            rot_id = f'{orig_id}-rotation{rot_idx}'

            hdf_subj = file_h5[split_id].create_group(rot_id)

            for vol, is_binary in zip(['csf_volume', 'exclude_volume', 'gm_volume', 'include_volume',
                                       'input_volume', 'interface_volume', 'peaks_volume', 'wm_volume'],
                                      [True, True, True, True, False, True, False, True]):
                hdf_input_volume = hdf_subj.create_group(vol)
                hdf_input_volume.attrs['vox2rasmm'] = file_h5[split_id][orig_id][vol].attrs['vox2rasmm']

                data = file_h5[split_id][orig_id][vol]['data'][:]

                orig_ddim = data.ndim
                if orig_ddim == 3:
                    data = data[..., None]

                # pad data to cube
                data = pad_data(data)

                if vol == 'input_volume':
                    signal_end_idx = o3.Irreps(in_irreps).dim
                    data_r, shift, rmat = rotate_data([alpha, beta],
                                                      data[...,
                                                      :signal_end_idx],
                                                      is_binary,
                                                      in_irreps=in_irreps,
                                                      return_trafo=True)

                    mask_r = rotate_data([alpha, beta], data[..., signal_end_idx:], is_binary=True)
                    data_r = np.concatenate([data_r, mask_r], axis=-1)
                else:
                    data_r = rotate_data([alpha, beta], data, is_binary)

                if orig_ddim == 3:
                    data_r = data_r[..., 0]

                # save out for debugging
                if vol == 'input_volume':
                    img_out = nib.Nifti1Image(data_r[..., :signal_end_idx],
                                          hdf_input_volume.attrs['vox2rasmm'])
                    nib.save(img_out, path_h5.replace('ismrm2015_lmax4-e3nn.hdf5', f'{rot_id}-{vol}.nii.gz'))

                hdf_input_volume.create_dataset('data', data=data_r)

            # process also the reference mask
            mask = nib.load(file_mask_in)
            mdata = mask.get_fdata()
            mdata = pad_data(mdata[..., None])
            mdata_r = rotate_data([alpha, beta], mdata, is_binary=True)
            mask_r = nib.Nifti1Image(mdata_r[..., 0], mask.affine, header=mask.header)
            nib.save(mask_r, join(path_mask_out, f'{rot_id}_wm.nii.gz'))

        rot_params[f'{rot_id}'] = {'shift': shift, 'rmat': rmat}


        np.save(path_h5.replace('ismrm2015_lmax4-e3nn.hdf5', f'rotparams.npy'),
                rot_params)

if __name__ == '__main__':
    # ATTENTION: works only for fods in e3nn basis

    # copy and load dataset
    # TODO remove line below
    argv = ['fourtytwo',
            '/fabi_project/data/ttl_anat_priors/fabi_tests/ismrm2015/ismrm2015_lmax4-e3nn.hdf5',
            '/fabi_project/data/datasets-rotation/ismrm-lmax4/ismrm2015_lmax4-e3nn.hdf5',
            '/fabi_project/data/ttl_anat_priors/fabi_tests/ismrm2015/masks/ismrm2015_wm.nii.gz',
            '/fabi_project/data/datasets-rotation/ismrm-lmax4/masks'
            ]

    dst_h5 = argv[2]
    file_rmask_in = argv[3]
    path_rmask_out = argv[4]
    shutil.copyfile(argv[1], dst_h5)

    # in_irreps_ = "1x0e+1x2e+1x4e+1x6e"
    in_irreps_ = "1x0e+1x2e+1x4e"

    # loop over all rotations and store files to outputdir
    rotate_hdf5(dst_h5, 'ismrm2015', file_rmask_in, path_rmask_out, in_irreps=in_irreps_)



