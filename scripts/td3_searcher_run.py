import sys
from distutils.dir_util import copy_tree
from os import makedirs, path
from dataclasses import dataclass

from TrackToLearn.searchers import td3_searcher

@dataclass
class FilePaths:
    dset_folder = '/fabi_project/data/ttl_anat_priors'
    work_dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    ttl_root = '/home/fabi/remote_TractoRL'
    test_subject_id = 'fibercup'
    subject_id = 'fibercup'
    expe_folder = path.join(dset_folder, 'experiments')
    work_expe_folder = path.join(work_dset_folder, 'experiments')
    scoring_data = path.join(dset_folder, 'raw_fabi_2024', test_subject_id, 'scoring_data')
    copy_tree(path.join(dset_folder, 'raw_fabi_2024', subject_id),
              path.join(work_dset_folder, 'raw_fabi_2024', subject_id))
    dataset_file = path.join(work_dset_folder, 'raw_fabi_2024', subject_id,
                             subject_id + '.hdf5') # -e3nn
    test_dataset_file = path.join(work_dset_folder, 'raw_fabi_2024', test_subject_id,
                                  test_subject_id + '.hdf5') # -e3nn
    test_reference_file = path.join(work_dset_folder, 'raw_fabi_2024', test_subject_id,
                                    'masks', test_subject_id + '_wm.nii.gz')

fp = FilePaths()

if not path.exists(fp.work_dset_folder):
    makedirs(fp.work_dset_folder)

print("Transfering data to working folder...")
if not path.exists(path.join(fp.work_dset_folder, 'raw_fabi_2024', fp.subject_id)):
    makedirs(path.join(fp.work_dset_folder, 'raw_fabi_2024', fp.subject_id))

experiment = 'TD3FiberCupFabi'
# ID=$(date +"%F-%H_%M_%S")
expe_id = "fibercup-td3"
# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
    curr_dset_folder = path.join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    # FILEPATHS AND IDS
    argv = [f'{fp.ttl_root}/TrackToLearn/trainers/td3_train.py',
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
    argv += [f'--neighborhood_mode=star',
             f'--max_ep={1000}',                # choosen empirically
             f'--log_interval={25}',
             f'--action_std={0.2}',
             f'--rng_seed={rng_seed}',
             f'--npv={2}',                    # number of seeds per voxel
             f'--theta={30}',
             f'--prob={0.0}',
             f'--use_gpu',
             f'--run_tractometer',
             f'--interface_seeding',
             f'--min_length={20}',
             f'--max_length={150}',
             f'--use_comet',
             f'--n_actor={2048}',              # directly affects batchsize
             f'--n_dirs={2}',                     # no previous directions
             f'--lr={0.0005}',
             # f'--complexity_weighting={1.0}',
             # f'--alignment_weighting={1}',
             # f'--straightness_weighting={.1}'  
             # f'--init_dirs_from_peaks'          #
             f'--so3_actor_input_irrep={"1x0e+1x2e+1x4e+1x6e"}'
             ]

    # HYPERPARAMETERS TO OPTIMIZE
    config = {
        "algorithm": "bayes",
        "parameters": {
            "add_neighborhood": {
                "type": "discrete",
                "values": [0.75],
            },
            "gamma": {
                "type": "discrete",
                "values": [0.9]},
        },
        "spec": {
            "metric": "Reward",
            "objective": "maximize",
            "seed": rng_seed,
        },
    }


    sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

    # CALL
    td3_searcher.main(config)

