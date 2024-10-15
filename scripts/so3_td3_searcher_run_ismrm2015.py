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
    test_subject_id = 'ismrm2015'
    subject_id = 'ismrm2015'
    expe_folder = path.join(dset_folder, 'experiments')
    work_expe_folder = path.join(work_dset_folder, 'experiments')
    scoring_data = path.join(dset_folder, test_subject_id, 'scoring_data')
    copy_tree(path.join(dset_folder, subject_id),
              path.join(work_dset_folder, subject_id))
    dataset_file = path.join(work_dset_folder, subject_id,
                             subject_id + '-e3nn.hdf5')
    test_dataset_file = path.join(work_dset_folder, test_subject_id,
                                  test_subject_id + '-e3nn.hdf5')
    test_reference_file = path.join(work_dset_folder, test_subject_id,
                                    'masks', test_subject_id + '_wm.nii.gz')

fp = FilePaths()

if not path.exists(fp.work_dset_folder):
    makedirs(fp.work_dset_folder)

print("Transfering data to working folder...")
if not path.exists(path.join(fp.work_dset_folder, fp.subject_id)):
    makedirs(path.join(fp.work_dset_folder, fp.subject_id))

experiment = 'SO3TD3-ISMRM2015'
# ID=$(date +"%F-%H_%M_%S")
expe_id = "ismrm2015-td3-so3"
# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
    curr_dset_folder = path.join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    # FILEPATHS AND IDS
    argv = [f'{fp.ttl_root}/TrackToLearn/trainers/so3_td3_train.py',
            curr_dset_folder,
            experiment,
            expe_id,
            fp.dataset_file,
            fp.subject_id,
            fp.test_dataset_file,
            fp.test_subject_id,
            fp.test_reference_file,
            fp.scoring_data]

    # STATIC ARGUMENTS (don't change in hyperopt)
    argv += [f'--max_ep={10000}',                # choosen empirically
             f'--log_interval={10}',
             f'--action_std={0.1}',
             f'--rng_seed={rng_seed}',
             f'--npv={1}',                    # number of seeds per voxel
             f'--theta={30}',
             f'--prob={0.01}',
             f'--use_gpu',
             f'--run_tractometer',
             # f'--interface_seeding',
             f'--min_length={20}',
             f'--max_length={150}',
             f'--use_comet',
             # f'--n_actor={2048}',              # directly affects batchsize
             f'--n_actor={256}',              # directly affects batchsize
             f'--n_dirs={2}',                     # no previous directions
             f'--complexity_weighting={1.0}',
             # f'--alignment_weighting={1}',
             # f'--straightness_weighting={.1}'  
             # f'--init_dirs_from_peaks'          #
             ]

    # MODEL ARGUMENS
    # SO3 SPECIFIC
    # critic model
    argv +=[# f'--so3_critic_lr={1e-3}',
            f'--so3_critic_hidden_dims={"1024-1024"}',
            # f'--so3_actor_lr={1e-1}',
            # f'--so3_spheredir',
            # f'--so3_actor_hidden_lmax={"4x0e+4x1o+4x2e+4x4e+2x6e"}',
            # f'--so3_actor_hidden_depth={3}',
            # f'--add_neighborhood={2}',
            ]

    # HYPERPARAMETERS TO OPTIMIZE
    config = {
        "algorithm": "bayes",
        "parameters": {
            "add_neighborhood": {
                "type": "discrete",
                "values": [1],
            },
            "gamma": {
                "type": "discrete",
                "values": [0.90]},
            "angle_penalty_factor": {
                "type": "discrete",
                "values": [1]},
            "so3_critic_lr": {
                "type": "discrete",
                "values": [1e-4, 1e-3]
            },
            "so3_actor_lr": {
                "type": "discrete",
                "values": [1e-5]  # 1e-1
            },
            "so3_actor_hidden_depth": {
                "type": "discrete",
                "values": [5]
            },
            "so3_actor_hidden_lmax": {
                "type": "categorical",
                "values":[
                    # "4x0e+4x1o+4x2e+4x4e+2x6e",
                    "4x0e+4x1o+4x2e+4x3o+4x4e",
                    ],
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
