import sys
from distutils.dir_util import copy_tree
from os import makedirs, path
from dataclasses import dataclass

from TrackToLearn.searchers import so3_td3_searcher

# parse variational positional args
exp1_argv = []

rot_state = 'nors'
rot_equiv = 'noreq'

assert len(sys.argv) == 1 or len(sys.argv) == 3, (
    'usage: <name.py> <bool: so3_neighb_from_peaks> <bool: so3_rotate_equiv_graph>')
if len(sys.argv) == 3:
    if sys.argv[1].lower() == 'true':
        rot_state = 'rs'
        exp1_argv += ['--so3_neighb_from_peaks']
    if sys.argv[2].lower() == 'true':
        rot_equiv = 'req'
        exp1_argv += ['--so3_rotate_equiv_graph']


@dataclass
class FilePaths:
    # dset_folder = '/proj/berzelius-2024-159/1_data/data/datasets'
    # work_dset_folder = '/proj/berzelius-2024-159/2_experiments'
    # ttl_root = '/home/x_fabsi/0_projectdir/p3_TractoRL'
    dset_folder = '/fabi_project/data/datasets'
    work_dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    ttl_root = '/home/fabi/remote_TractoRL'
    test_subject_id = 'fibercup_lmax4'
    subject_id = 'fibercup'
    work_expe_folder = path.join(work_dset_folder, 'experiments')
    scoring_data = path.join(dset_folder, test_subject_id, 'scoring_data')
    copy_tree(path.join(dset_folder, test_subject_id),
              path.join(work_dset_folder, test_subject_id))
    dataset_file = path.join(work_dset_folder, test_subject_id,
                             subject_id + '_lmax4-e3nn.hdf5')
    test_dataset_file = path.join(work_dset_folder, test_subject_id,
                                  subject_id + '_lmax4-e3nn.hdf5')
    test_reference_file = path.join(work_dset_folder, test_subject_id,
                                    'masks', subject_id + '_wm.nii.gz')
    so3_spheredir = '/fabi_project/sphere'

fp = FilePaths()

experiment = 'Exp2'
# ID=$(date +"%F-%H_%M_%S")
expe_id = f"exp2-so3-fodf4-{rot_state}-{rot_equiv}"
# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    # argv = list()
    curr_expe_folder = path.join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    # FILEPATHS AND IDS
    argv = [f'{fp.ttl_root}/TrackToLearn/trainers/so3_td3_train.py',
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
             f'--log_interval={10}',
             f'--action_std={0.1}',
             f'--rng_seed={rng_seed}',
             f'--npv={1}',   # 10 !!                 # number of seeds per voxel
             f'--theta={60}',
             f'--prob={0.0}',
             f'--use_gpu',
             f'--run_tractometer',
             f'--use_comet',
             f'--n_actor={2**8}',              # directly affects batchsize
             ]

    # MODEL ARGUMENS
    # SO3 SPECIFIC
    # critic model
    argv +=[
            f'--so3_critic_hidden_dims={"1024-1024"}',
            f'--so3_spheredir={fp.so3_spheredir}',
            f'--so3_actor_input_irrep={"1x0e+1x2e+1x4e"}',
            ]

    # exp1 specific
    argv += exp1_argv

    # HYPERPARAMETERS TO OPTIMIZE
    config = {
        "algorithm": "grid",
        "parameters": {
            "add_neighborhood": {
                "type": "discrete",
                "values": [1.],
            },
            "n_dirs": {
                "type": "discrete",
                "values": [2]},
            "gamma": {
                "type": "discrete",
                "values": [0.9]},
            "so3_critic_lr": {
                "type": "discrete",
                "values": [1e-3]  # 1e-5
            },
            "so3_actor_lr": {
                "type": "discrete",
                "values": [1e-3]  # 1e-5
            },
            "so3_actor_hidden_depth": {
                "type": "discrete",
                "values": [3]
            },
            "so3_actor_hidden_lmax": {
                "type": "discrete",
                "values": [3],  # [2, 3, 4],
            },
        },
        "spec": {
            "metric": "Reward",
            "objective": "maximize",
            "seed": rng_seed,
        },
    }

    sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

    # CALL
    so3_td3_searcher.main(config)

