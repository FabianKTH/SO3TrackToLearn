import os.path
from os.path import join, exists
from os import makedirs
from distutils.dir_util import copy_tree
from TrackToLearn.trainers import so3_td3_train, td3_train
import sys

DATASET_FOLDER='/fabi_project/data/ttl_anat_priors'
WORK_DATASET_FOLDER='/fabi_project/data/ttl_anat_priors/fabi_tests'
TTL_ROOT='/home/fabi/TractoRL'

if not exists(WORK_DATASET_FOLDER):
    makedirs(WORK_DATASET_FOLDER)

TEST_SUBJECT_ID='fibercup'
SUBJECT_ID='fibercup'
EXPERIMENTS_FOLDER=join(DATASET_FOLDER, 'experiments')
WORK_EXPERIMENTS_FOLDER=join(WORK_DATASET_FOLDER, 'experiments')
SCORING_DATA=join(DATASET_FOLDER, 'raw_fabi_2024', TEST_SUBJECT_ID, 'scoring_data')

print("Transfering data to working folder...")
if not exists(join(WORK_DATASET_FOLDER, 'raw_fabi_2024', SUBJECT_ID)):
  makedirs(join(WORK_DATASET_FOLDER, 'raw_fabi_2024', SUBJECT_ID))

copy_tree(join(DATASET_FOLDER, 'raw_fabi_2024', SUBJECT_ID),
          join(WORK_DATASET_FOLDER, 'raw_fabi_2024', SUBJECT_ID))

dataset_file=join(WORK_DATASET_FOLDER, 'raw_fabi_2024', SUBJECT_ID, SUBJECT_ID + '.hdf5')
test_dataset_file=join(WORK_DATASET_FOLDER, 'raw_fabi_2024', TEST_SUBJECT_ID, TEST_SUBJECT_ID + '.hdf5')
test_reference_file=join(WORK_DATASET_FOLDER, 'raw_fabi_2024', TEST_SUBJECT_ID, 'masks', TEST_SUBJECT_ID + '_wm.nii.gz')

max_ep=500 # (1500) Chosen empirically
log_interval=25 # Log at n steps
lr=0.0005 # (0.0005) Learning rate
gamma=0.9 # Gamma for reward discounting
alpha=0.2

valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # (2) Seed per voxel
max_angle=30 # (30) Maximum angle for streamline curvature

EXPERIMENT='TD3FiberCup'

# ID=$(date +"%F-%H_%M_%S")
ID="_debug_log5-get_prev"

# seeds=(1111 2222 3333 4444 5555)
seeds=['1111']

for rng_seed in seeds:

  DEST_FOLDER=join(WORK_EXPERIMENTS_FOLDER, EXPERIMENT, ID, rng_seed)

  argv = [f'{TTL_ROOT}/TrackToLearn/trainers/td3_train.py',
   DEST_FOLDER,
   EXPERIMENT,
   ID,
   dataset_file,
   SUBJECT_ID,
   test_dataset_file,
   TEST_SUBJECT_ID,
   test_reference_file,
   SCORING_DATA,
   f'--max_ep={max_ep}',
   f'--log_interval={log_interval}',
   f'--lr={lr}',
   f'--gamma={gamma}',
   f'--action_std={alpha}',
   f'--rng_seed={rng_seed}',
   f'--npv={n_seeds_per_voxel}',
   f'--theta={max_angle}',
   f'--prob={valid_noise}',
   f'--use_gpu',
   f'--run_tractometer',
   f'--interface_seeding',
   f'--use_comet',
    f'--min_length=20',
    f'--max_length=200'
    ]

  sys.argv = argv # hack, replace sys.argv to make the follow script accept the args

  # call
  td3_train.main()

  if not exists(f'EXPERIMENTS_FOLDER/{EXPERIMENT}'):
    makedirs(f'EXPERIMENTS_FOLDER/{EXPERIMENT}')
  if not exists(f'EXPERIMENTS_FOLDER/{EXPERIMENT}/{ID}'):
    makedirs(f'EXPERIMENTS_FOLDER/{EXPERIMENT}/{ID}')

  copy_tree(DEST_FOLDER, f'{EXPERIMENTS_FOLDER}/{EXPERIMENT}/{ID}/')
