from os.path import join
from dataclasses import dataclass
import numpy as np

from TrackToLearn.csd_tracto import closespeak


@dataclass
class FilePaths:
    # dset_folder = '/proj/berzelius-2024-159/1_data/data/datasets'
    # work_dset_folder = '/proj/berzelius-2024-159/2_experiments'
    # ttl_root = '/home/x_fabsi/0_projectdir/p3_TractoRL'
    scoring_data = ('/fabi_project/data/ttl_anat_priors/fabi_tests/ismrm2015'
                    '/scoring_data')
    dset_folder = '/fabi_project/data'
    ttl_root = '/home/fabi/remote_TractoRL'
    work_expe_folder = '/fabi_project/models'

    test_subject_id = 'ismrm2015'
    subject_id = 'ismrm2015'
    # scoring_data = path.join(work_dset_folder, test_subject_id,
    # 'scoring_data')

    dataset_file = join(dset_folder, 'datasets-rotation', 'ismrm-lmax4',
                        test_subject_id + '_lmax4-e3nn.hdf5')
    reference_folder = join(dset_folder, 'datasets-rotation', 'ismrm-lmax4',
                            'masks')

    so3_spheredir = '/fabi_project/sphere'


fp = FilePaths()
experiment = 'CSD-CLOSEST'
expe_id = "Exp3-csd-closest"
no_rots = 12
subject_ids = [f'{fp.test_subject_id}-rotation{rot_idx}' for rot_idx in
               range(no_rots)]
# add original (no rotations)
# subject_ids = [f'{fp.test_subject_id}'] + subject_ids

# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
    curr_expe_folder = join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    rotation_parameters = np.load(
            fp.reference_folder.replace('masks', 'rotparams.npy'),
            allow_pickle=True).item()

    result_scores = []

    for subject_id in subject_ids:
        reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')

        kwargs = {
            'dataset_file': fp.dataset_file,
            'split': 'validation',
            'subject_id': subject_id,
            'interface_seeding': False,
            'npv': 1,  # 300
            'step_size': .75,
            'ref_file': reference_file
        }

        tracking_model = closespeak.ClosestPeakExperiment(
            kwargs
        )
        tracking_model.run()


        # TODO!! rotate tractogram back in order to score submission

        pass
