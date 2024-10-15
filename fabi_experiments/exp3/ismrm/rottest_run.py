from dataclasses import dataclass
from os.path import join
import h5py
import numpy as np

from TrackToLearn.utils.rotation import roto_transl, roto_transl_inv, rotate_irr
from TrackToLearn.utils.seeds import seeds_from_maskfile

from TrackToLearn.datasets.utils import SubjectData
from TrackToLearn.environments.interpolation import \
    interpolate_volume_at_coordinates


def interp(x, xcoord):
    return interpolate_volume_at_coordinates(x,
                                      xcoord.astype(np.float32))


@dataclass
class FilePaths:
    dset_folder = '/fabi_project/data'
    ttl_root = '/home/fabi/remote_TractoRL'
    work_expe_folder = '/fabi_project/models'

    test_subject_id = 'ismrm2015'
    subject_id = 'ismrm2015'
    # scoring_data = path.join(work_dset_folder, test_subject_id,
    # 'scoring_data')

    dataset_file = join(dset_folder, 'datasets-rotation', 'ismrm-lmax4-fix',
                        test_subject_id + '_fix_lmax4.hdf5')
    reference_folder = join(dset_folder, 'datasets-rotation', 'ismrm-lmax4-fix',
                            'masks')

    training_seeding_mask = ('/fabi_project/data/datasets/ismrm2015/masks'
                             '/ismrm2015_wm.nii.gz')

    so3_spheredir = '/fabi_project/sphere'


fp = FilePaths()
experiment = 'Exp3'
expe_id = f"rotation_sampling"

no_rots = 6
subject_ids = [f'{fp.test_subject_id}-rotation{rot_idx}' for rot_idx in
               range(no_rots)]

curr_expe_folder = join(fp.work_expe_folder, experiment, expe_id)
npv = 1

with h5py.File(fp.dataset_file, 'r') as h5_file:
    subj_data_norot = SubjectData.from_hdf_subject(
            h5_file['validation'],
            'ismrm2015')

tracking_seeds = seeds_from_maskfile(fp.training_seeding_mask, npv,
                                     int('42')
                                     )

# sample at original
signal_norot = interp(subj_data_norot.input_dv.data[..., :-1], tracking_seeds)

# !! TODO remove next line, testing
# subject_ids= ['ismrm2015-rotation5']

for subject_id in subject_ids:
    reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')
    model_folder = join(fp.work_expe_folder, experiment, expe_id)

    with  h5py.File(fp.dataset_file, 'r') as h5_file:
        subj_data_rot = SubjectData.from_hdf_subject(
                h5_file['validation'],
                subject_id)

    # load rotation matrix
    linear_mat = np.loadtxt(join(fp.reference_folder.replace('masks', subject_id + '.txt')))
    rmat = linear_mat[:3, :3]
    shift = linear_mat[-1, :3]

    # rotate preloaded tracking seeds
    seeds_rotated = roto_transl_inv(tracking_seeds, rmat, shift,
                                padshift=np.array([9., 0., 9.]))

    # sample at rotated
    signal_rot = interp(subj_data_rot.input_dv.data[..., :-1], seeds_rotated)
    signal_invrot = rotate_irr('1x0e+1x2e+1x4e', rmat.T, signal_rot,
                               return_np=True)

    pass


