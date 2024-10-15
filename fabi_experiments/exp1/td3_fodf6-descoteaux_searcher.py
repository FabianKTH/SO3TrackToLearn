import sys
from distutils.dir_util import copy_tree
from os import makedirs, path
from dataclasses import dataclass
from datetime import datetime


from TrackToLearn.searchers import td3_searcher

# parse variational positional args
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
    dset_folder = '/fabi_project/data/datasets'
    work_dset_folder = '/fabi_project/data/work-datasets'
    ttl_root = '/home/fabi/remote_TractoRL'
    work_expe_folder = '/fabi_project/experiments'

    test_subject_id = 'ismrm2015'
    subject_id = 'ismrm2015'
    scoring_data = path.join(dset_folder, test_subject_id, 'scoring_data')
    copy_tree(path.join(dset_folder, test_subject_id),
              path.join(work_dset_folder, test_subject_id))
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
    argv = [f'{fp.ttl_root}/TrackToLearn/trainers/td3_train.py',
            curr_expe_folder,
            experiment,
            expe_id,
            fp.dataset_file,
            fp.subject_id,
            fp.test_dataset_file,
            fp.test_subject_id,
            fp.test_reference_file,
            fp.scoring_data]

    # STATIC ARGUMENTS (don't change in hyperopt)
    argv += [f'--neighborhood_mode=star',
             f'--max_ep={250}',                # choosen empirically
             f'--log_interval={25}',           # choose lower for final runs
             f'--action_std={0.4}',            # this is the action noise $\sigma$
             f'--rng_seed={rng_seed}',
             f'--npv={1}',                     # number of seeds per voxel
             f'--theta={60}',
             f'--prob={0.0}',                  # I think this is valid noise
             f'--use_gpu',
             f'--run_tractometer',
             f'--use_comet',
             f'--n_actor={2**12}', # 12 before              # directly affects batchsize
             f'--lr={0.00001}',
             ]

    # MODEL ARGUMENS
    # SO3 SPECIFIC
    argv +=[
            f'--so3_actor_input_irrep={"1x0e+1x2e+1x4e+1x6e"}',
            ]

    # exp1 specific
    argv += exp1_argv

    # HYPERPARAMETERS TO OPTIMIZE
    config = {
        "algorithm": "grid",
        "parameters": {
            "n_dirs": {
                "type": "discrete",
                "values": [2]},
            "gamma": {
                "type": "discrete",
                "values": [0.85]},
            },
        "spec": {
            "metric": "Reward",
            "objective": "maximize",
            "seed": rng_seed,
        }
    }

    sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

    # CALL
    td3_searcher.main(config)

