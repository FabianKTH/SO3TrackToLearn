import sys
from distutils.dir_util import copy_tree
from os import makedirs, path
from dataclasses import dataclass

from TrackToLearn.searchers import so3_td3_searcher

@dataclass
class FilePaths:
    dset_folder = '/fabi_project/data/datasets'
    work_dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    ttl_root = '/home/fabi/remote_TractoRL'
    test_subject_id = 'fibercup_peaks'
    subject_id = 'fibercup'
    work_expe_folder = path.join(work_dset_folder, 'experiments')
    scoring_data = path.join(dset_folder, test_subject_id, 'scoring_data')
    copy_tree(path.join(dset_folder, test_subject_id),
              path.join(work_dset_folder, test_subject_id))
    dataset_file = path.join(work_dset_folder, test_subject_id,
                             subject_id + '_peaks.hdf5')
    test_dataset_file = path.join(work_dset_folder, test_subject_id,
                                  subject_id + '_peaks.hdf5')
    test_reference_file = path.join(work_dset_folder, test_subject_id,
                                    'masks', subject_id + '_wm.nii.gz')
    so3_spheredir = '/fabi_project/sphere'

fp = FilePaths()

# if not path.exists(fp.work_dset_folder):
#     makedirs(fp.work_dset_folder)

# print("Transfering data to working folder...")
# if not path.exists(path.join(fp.work_dset_folder, fp.test_subject_id)):
#     makedirs(path.join(fp.work_dset_folder, fp.test_subject_id))

experiment = 'SO3TD3FiberCupPeaksExp1'
# ID=$(date +"%F-%H_%M_%S")
expe_id = "fibercup-td3-so3-peaks1"
# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
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
             f'--max_ep={100}',                # choosen empirically
             f'--log_interval={50}',
             f'--action_std={0.25}',
             f'--rng_seed={rng_seed}',
             f'--npv={10}',                    # number of seeds per voxel
             f'--theta={60}',
             f'--prob={0.0}',
             f'--use_gpu',
             f'--run_tractometer',
             f'--use_comet',
             f'--n_actor={2**14}',              # directly affects batchsize
             # f'--complexity_weighting={1.0}',
             # f'--alignment_weighting={1}',
             # f'--straightness_weighting={.1}'  
             ]

    # MODEL ARGUMENS
    # SO3 SPECIFIC
    # critic model
    argv +=[# f'--so3_critic_lr={1e-3}',
            f'--so3_critic_hidden_dims={"1024-1024"}',
            f'--so3_spheredir={fp.so3_spheredir}',
            f'--so3_actor_input_irrep={"10x1o"}',
            ]

    # HYPERPARAMETERS TO OPTIMIZE
    config = {
        "algorithm": "bayes",
        "parameters": {
            "add_neighborhood": {
                "type": "discrete",
                "values": [0.5, 1., 2.],
            },
            "n_dirs": {
                "type": "discrete",
                "values": [1, 2, 4]},
            "gamma": {
                "type": "discrete",
                "values": [0.75]},
            "so3_critic_lr": {
                "type": "discrete",
                "values": [1e-4, 1e-5]  # 1e-5
            },
            "so3_actor_lr": {
                "type": "discrete",
                "values": [1e-4, 1e-5]  # 1e-5
            },
            "so3_actor_hidden_depth": {
                "type": "discrete",
                "values": [4]
            },
            "so3_actor_hidden_lmax": {
                "type": "discrete",
                "values": [2, 3],  # [2, 3, 4],
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

