import sys
from os import path
from dataclasses import dataclass

import ttl_validation

exp1_argv = []
rot_state = 'nors'

if len(sys.argv) == 1:
    exp1_argv += [f'--rng_seed={1111}']
    seeds = ['1111']
elif len(sys.argv) == 3:
    so3_degree = int(sys.argv[2]) # has no effect here
    exp1_argv += [f'--rng_seed={int(sys.argv[1])}']
    seeds = [sys.argv[1]]
else:
    raise ValueError('usage: <name.py> <SEEEED> <L_MAX>')

exp1_argv += [f'--n_actor={2**12}']

@dataclass
class FilePaths:
    # dset_folder = '/proj/berzelius-2024-159/1_data/data/datasets'
    # work_dset_folder = '/proj/berzelius-2024-159/2_experiments'
    # ttl_root = '/home/x_fabsi/0_projectdir/p3_TractoRL'
    work_dset_folder = '/fabi_project/data/work-datasets'
    ttl_root = '/home/fabi/remote_TractoRL'
    work_expe_folder = '/fabi_project/experiments'

    test_subject_id = 'ismrm2015'
    subject_id = 'ismrm2015'
    scoring_data = path.join(work_dset_folder, test_subject_id, 'scoring_data')
    dataset_file = path.join(work_dset_folder, test_subject_id,
                             subject_id + '.hdf5')
    test_dataset_file = path.join(work_dset_folder, test_subject_id,
                                  subject_id + '.hdf5')
    test_reference_file = path.join(work_dset_folder, test_subject_id,
                                    'masks', subject_id + '_wm.nii.gz')
    so3_spheredir = '/fabi_project/sphere'

fp = FilePaths()

experiment = 'Exp1'
# ID=$(date +"%F-%H_%M_%S")
expe_id = f"exp1-td3-fodf4-{rot_state}"
# seeds=(1111 2222 3333 4444 5555)

for rng_seed in seeds:
    # argv = list()
    curr_expe_folder = path.join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    # FILEPATHS AND IDS
    argv = [f'{fp.ttl_root}/rrl_validation',
            curr_expe_folder,
            experiment,
            expe_id,
            fp.dataset_file,
            fp.subject_id,
            fp.test_reference_file,
            path.join(fp.work_expe_folder, experiment, expe_id, rng_seed,
                      'model'),
            path.join(fp.work_expe_folder, experiment, expe_id, rng_seed,
                      'model', 'hyperparameters.json'),
            f'--npv={1}',
            # f'--n_actor={8000}',
            # f'--min_length={20}',
            # f'--max_length={200}',  # TODO : 200
            f'--use_gpu',
            # f'--remove_invalid_streamlines',
            f'--scoring_data={fp.scoring_data}']

    # exp1 specific
    argv += exp1_argv
    sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

    # CALL
    ttl_validation.main()

