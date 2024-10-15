import sys
from dataclasses import dataclass
from os import path, listdir

import nibabel as nib
import pandas as pd
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs

import ttl_validation

exp1_argv = []
eval_trac_scores = False
exp1_argv += [f'--n_actor={2 ** 13}']
seeds = ['1111', '2222', '3333', '4444', '5555']

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
                             subject_id + '_fix_lmax4.hdf5')
    test_dataset_file = path.join(work_dset_folder, test_subject_id,
                                  subject_id + '_fix_lmax4.hdf5')
    test_reference_file = path.join(work_dset_folder, test_subject_id,
                                    'masks', subject_id + '_wm.nii.gz')
    so3_spheredir = '/fabi_project/sphere'


fp = FilePaths()

experiment = 'Exp1'
expe_id = f"exp1-td3-final_nodirinstate_first50"

for rng_seed in seeds:
    # argv = list()
    curr_expe_folder = path.join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    score_record = list()

    # loop over all stored models
    for fname in listdir(path.join(curr_expe_folder, 'model')):
        if fname.endswith('_actor.pth'):
            pretrained_actor = fname.replace('_actor.pth', '')
            print(f'found model {pretrained_actor}')

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
                    f'--min_length={20}',
                    f'--max_length={200}',  # TODO : 200
                    f'--use_gpu',
                    f'--remove_invalid_streamlines',
                    f'--scoring_data={fp.scoring_data}']

            # exp1 specific
            argv += exp1_argv
            sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

            # CALL
            tracto_dict, reward, equiv = ttl_validation.main(return_tracto=True,
                                                             pretrained_model=pretrained_actor,
                                                             validate=True)

            scores = {}
            nib.streamlines.save(tracto_dict['tractogram'],
                                 tracto_dict['filename'],
                                 header=tracto_dict['header'])

            basic_bundles_attribs = load_attribs(path.join(fp.scoring_data,
                                                           'gt_bundles_attributes.json'))
            if eval_trac_scores:
                scores = score_submission(tracto_dict['filename'],
                                          fp.scoring_data,
                                          basic_bundles_attribs,
                                          compute_ic_ib=True)

            scores['actor_name'] = pretrained_actor
            scores['reward'] = reward
            scores['equiv'] = equiv

            print(scores)
            score_record.append(scores)

    df_ = pd.DataFrame.from_records(score_record, index='actor_name')
    df_.to_csv(path.join(curr_expe_folder, 'all_scores.csv'))
