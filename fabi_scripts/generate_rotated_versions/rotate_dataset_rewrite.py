import shutil
from os.path import join

import h5py
import nibabel as nib
import numpy as np
import torch
from e3nn import o3
from tqdm import tqdm

from TrackToLearn.environments.interpolation import \
    interpolate_volume_at_coordinates
from TrackToLearn.utils.rotation import get_ico_points, rotate_irr
from TrackToLearn.utils.torch_shorthands import tnp, tt


class DsetRotator:

    data_info = [('csf_volume',         True, '1x0e'),
                 ('exclude_volume',     True, '1x0e'),
                 ('gm_volume',          True, '1x0e'),
                 ('include_volume',     True, '1x0e'),
                 ('input_volume',       False, '1x0e+1x2e+1x4e'),
                 ('interface_volume',   True, '1x0e'),
                 ('peaks_volume',       False, '5x1o'),
                 ('wm_volume',          True, '1x0e')]


    def __init__(self,
                 in_dset_h5,
                 out_dset_h5,
                 out_path,
                 reference_file,
                 id,
                 split_id='validation',
                 angles=None,
                 debug=False):

        self.debug = debug
        self.id = id
        self.split_id = split_id

        # make target file from input
        shutil.copyfile(in_dset_h5, out_dset_h5)
        self.out_dset_h5 = out_dset_h5
        self.out_path = out_path

        # load mask to setup grid and dims
        self.ref_file = nib.load(reference_file)
        self.ref_data = self.ref_file.get_fdata()
        self.dimxyz = [max(self.ref_data.shape[:3])] * 3 # size padded
        self._init_grid()
        self.angles = angles if angles is not None else self._init_angles()


    def _init_grid(self):
        # make grid
        dimx, dimy, dimz = self.dimxyz
        x, y, z = np.meshgrid(
                np.linspace(0, dimx - 1, dimx),
                np.linspace(0, dimy - 1, dimy),
                np.linspace(0, dimz - 1, dimz)
                , indexing='ij')
        idx = np.stack([x, y, z], axis=-1)

        idx = idx.reshape(-1, 3)

        # center idx
        self.centershift = np.array([dimx / 2, dimy / 2, dimz / 2])[None]
        idx -=self.centershift

        self.idx = idx

    @staticmethod
    def _init_angles():
        xyz = get_ico_points(subdiv=3)
        alphas, betas = o3.xyz_to_angles(tt(xyz))

        return [(alp, bet) for (alp, bet) in zip(alphas, betas)]


    def _pad_data(self, x):
        max_d = self.dimxyz[0]
        dimx, dimy, dimz = x.shape[:3]

        p_beforex, p_beforey, p_beforez = ((max_d - dimx)//2,
                                           (max_d - dimy)//2,
                                           (max_d - dimz)//2)
        p_afterx, p_aftery, p_afterz = (max_d - (p_beforex + dimx),
                                        max_d - (p_beforey + dimy),
                                        max_d - (p_beforez + dimz))
        pad_width = ((p_beforex, p_afterx),
                     (p_beforey, p_aftery),
                     (p_beforez, p_afterz),
                     (0, 0))

        x_padded = np.pad(x, pad_width, mode='constant')  # const 0 by default

        return x_padded, (p_beforex, p_beforey, p_beforez)

    @staticmethod
    def _get_rmat(angles):
        return o3.angles_to_matrix(angles[0], angles[1],
                                   torch.Tensor([0.]).cuda())[0]


    def rotate(self, x, rmat, irreps, is_mask):
        irr_dim = x.shape[-1]

        # rotate volume coordinates
        idx_r = np.einsum('xi,ij->xj', self.idx, tnp(rmat))  # tnp(rmat.T))

        # move to old center
        idx_r += (np.array(self.dimxyz)/2)[None]

        # sample volume
        spline_order = 0 if is_mask else 1
        x_r = interpolate_volume_at_coordinates(x, idx_r,
                                                   order=spline_order)

        # rotate also the sph/vectors
        if not is_mask:
            x_r = rotate_irr(irreps, rmat, x_r)

        x_r = x_r.reshape(*self.dimxyz, irr_dim)

        return x_r

    def rotate_data(self, x, vol_name, irreps, is_mask, angles):
        orig_dim = len(x.shape)
        if orig_dim == 3:
            x = x[..., None]

        rmat = self._get_rmat(angles)
        # shift = np.array(self.dimxyz)/2

        x, _ = self._pad_data(x)
        if vol_name == 'input_volume':
            signal_end = o3.Irreps(irreps).dim
            x_r = self.rotate(x[..., :signal_end], rmat, irreps, is_mask)
            mask_r = self.rotate(x[..., signal_end:], rmat, irreps,
                                 is_mask=True)
            x_r = np.concatenate([x_r, mask_r], axis=-1)
        else:
            x_r = self.rotate(x, rmat, irreps, is_mask)

        if orig_dim == 3:
            x_r = x_r[..., 0]

        return x_r


    def run(self):
        rot_params = {}

        for r_idx, (alpha, beta) in tqdm(enumerate(self.angles)):
            with h5py.File(self.out_dset_h5, 'r+') as file_h5:

                rot_id = f'{self.id}-rotation{r_idx}'
                hdf_subj = file_h5[self.split_id].create_group(rot_id)

                # rotate all data volumes
                for vol_name, is_mask, irrep in self.data_info:

                    hdf_input_volume = hdf_subj.create_group(vol_name)
                    hdf_input_volume.attrs['vox2rasmm'] = \
                        file_h5[self.split_id][self.id][vol_name].attrs[
                            'vox2rasmm']

                    data = file_h5[self.split_id][self.id][vol_name]['data'][:]
                    data_r = self.rotate_data(data, vol_name, irrep,
                                              is_mask, (alpha, beta))

                    hdf_input_volume.create_dataset('data', data=data_r)

                    # save nifty files for debugging
                    if vol_name == 'input_volume' and self.debug:
                        self._debug_save_file(data_r[..., :-1],
                                              hdf_input_volume.attrs[
                                                  'vox2rasmm'],
                                              join(self.out_path,
                                                   f'{rot_id}-'
                                                   f'{vol_name}.nii.gz')
                                              )
                # process also reference mask
                ref_data_r = self.rotate_data(self.ref_data, 'reference',
                                              '1x0e', True, (alpha, beta))
                mask_nii = nib.Nifti1Image(ref_data_r, self.ref_file.affine,
                                           header=self.ref_file.header)
                nib.save(mask_nii, join(self.out_path, 'masks',
                                        f'{rot_id}_wm.nii.gz'))

                # write rot mat to file for mrtrix
                self.save_linearmat(alpha, beta, rot_id)

    def save_linearmat(self, alpha, beta, rot_id):
        rmat4x4 = np.zeros([4, 4])
        rmat4x4[-1, -1] = 1.  # by convention
        rmat4x4[-1, :3] = self.centershift
        rmat4x4[:3, :3] = tnp(self._get_rmat((alpha, beta)))
        np.savetxt(join(self.out_path, f'{rot_id}.txt'),
                   rmat4x4,
                   # fmt='%.4f')
                   )

    @staticmethod
    def _debug_save_file(x, aff, path_):
        img_out = nib.Nifti1Image(x, aff)
        nib.save(img_out, path_)


if __name__ == '__main__':
    # ATTENTION: works only for fods in e3nn basis

    # copy and load dataset
    # TODO remove line below
    argv = ['fourtytwo',
            '/fabi_project/data/ttl_anat_priors/fabi_tests/ismrm2015/ismrm2015_fix_lmax4.hdf5',
            '/fabi_project/data/datasets-rotation_subd2/ismrm-lmax4-fix'
            '/ismrm2015_fix_lmax4.hdf5',
            '/fabi_project/data/ttl_anat_priors/fabi_tests/ismrm2015/masks/ismrm2015_wm.nii.gz',
            '/fabi_project/data/datasets-rotation_subd2/ismrm-lmax4-fix'
            ]

    # angles for testing
    # betas = torch.zeros([6]).cuda() # torch.tensor([torch.pi/2.]*6).cuda()
    # alphas = torch.arange(0., torch.pi/4, torch.pi/24).cuda()
    # angles = [(alp, bet) for (alp, bet) in zip(alphas, betas)]

    rot_obj = DsetRotator(
            argv[1],
            join(argv[4], argv[1].split('/')[-1]),
            argv[4],
            argv[3],
            'ismrm2015',
            debug=True,
            # angles=angles
            )
    rot_obj.run()
