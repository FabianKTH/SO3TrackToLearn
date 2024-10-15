import sys
from distutils.dir_util import copy_tree
from os import makedirs, path
from os.path import join
from dataclasses import dataclass

from TrackToLearn.searchers import so3_td3_searcher
import ttl_validation

@dataclass
class FilePaths:
    dset_folder = '/fabi_project/data/ttl_anat_priors'
    ttl_root = '/home/fabi/remote_TractoRL'
    test_subject_id = 'ismrm2015'
    expe_folder = join(dset_folder, 'fabi_tests', 'experiments')
    scoring_data = join(dset_folder, 'raw_fabi_2024', test_subject_id, 'scoring_data')
    dataset_file = join(dset_folder, 'raw_fabi_2024', 'rotations', 'subdiv0',
                             test_subject_id + '-e3nn.hdf5')
    reference_folder = join(dset_folder, 'raw_fabi_2024/rotations/subdiv0/masks')


"""
/fabi_project/data/ttl_anat_priors/fabi_tests/experiments/
Exp2
pre_test
/fabi_project/data/datasets/ismrm2015/ismrm2015.hdf5
ismrm2015
/fabi_project/data/datasets/ismrm2015/masks/ismrm2015_wm.nii.gz
/home/fabi/TractoRL/example_model/SAC_Auto_ISMRM2015_WM
/home/fabi/TractoRL/example_model/SAC_Auto_ISMRM2015_WM/hyperparameters.json
--scoring_data
/fabi_project/data/ismrm2015_cleaned-ttl/scoring_data
"""


fp = FilePaths()
experiment = 'Exp3'
expe_id = "ismrm2015-td3-so3-hyperparam"
no_rots = 12
subject_ids = [f'{fp.test_subject_id}-rotation{rot_idx}' for rot_idx in range(no_rots)]
# add original (no rotations)
subject_ids = [f'{fp.test_subject_id}'] + subject_ids
# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
    curr_dest_folder = join(fp.expe_folder, experiment, expe_id, rng_seed)

    for subject_id in subject_ids:
        reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')

        # FILEPATHS AND IDS
        argv = [f'{fp.ttl_root}/ttl_validation.py',
                curr_dest_folder,
                experiment,
                expe_id,
                fp.dataset_file,
                subject_id,
                join(fp.reference_folder,  f'{subject_id}_wm.nii.gz'),

                join(curr_dest_folder, 'model'),
                join(curr_dest_folder, 'model', 'hyperparameters.json')
                ]

        # STATIC ARGUMENTS
        argv += [f'--npv={300}',
                 f'--n_actor={50000}',
                 f'--min_length={20}',
                 f'--max_length={200}',
                 f'--interface_seeding',
                 f'--use_gpu',
                 f'--remove_invalid_streamlines',
                 f'--init_dirs_from_peaks']

        sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

        # CALL
        ttl_validation.main()

        # TODO!! rotate tractogram back in order to score submission

        pass
