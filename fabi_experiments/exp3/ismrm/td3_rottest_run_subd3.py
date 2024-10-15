from dataclasses import dataclass
from os.path import join

import pandas as pd
import numpy as np

from TrackToLearn.utils.seeds import seeds_from_maskfile
from eval_rot_tracto import evalueate_rotated_tracking, evalueate_nonrotated_tracking


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
    # scoring_data = join(work_dset_folder, test_subject_id,
    # 'scoring_data')

    dataset_file = join(dset_folder, 'datasets-rotation_subd3', 'ismrm-lmax4-fix',
                        test_subject_id + '_fix_lmax4.hdf5')
    reference_folder = join(dset_folder, 'datasets-rotation_subd3', 'ismrm-lmax4-fix',
                            'masks')

    training_seeding_mask = ('/fabi_project/data/datasets/ismrm2015/masks'
                             '/ismrm2015_wm.nii.gz')

    so3_spheredir = '/fabi_project/sphere'


@dataclass
class ExperimentConfig:
    experiment = 'Exp3'
    expe_id = f"exp1-td3-fix_subd3"
    no_rots = 162
    base_id = f'ismrm2015'
    rotation_ids = [f'ismrm2015-rotation{rot_idx}'
                    for rot_idx in range(no_rots)]
    seeds = ['1111']
    npv = 1
    padshift = np.array([9., 0., 9.])
    n_actor = 2 ** 12


def run_experiment(fp, ec):

    # add original (no rotations)
    # rotation_ids = [f'{fp.test_subject_id}'] + rotation_ids

    for rng_seed in ec.seeds:
        curr_expe_folder = join(fp.work_expe_folder, ec.experiment, ec.expe_id, rng_seed)
        model_folder = join(fp.work_expe_folder, ec.experiment, ec.expe_id, rng_seed)
        result_scores = []

        tracking_seeds = seeds_from_maskfile(fp.training_seeding_mask,
                                             ec.npv,
                                             int(rng_seed)
                                             )

        # !! TODO remove, just for quicker debugging
        rng = np.random.default_rng(int(rng_seed))
        tracking_seeds = rng.choice(a=tracking_seeds, size=128, replace=False,
                                    axis=0)

        # first: run without rotation (base case)
        # FILEPATHS AND IDS
        argv = [f'{fp.ttl_root}/ttl_validation.py', curr_expe_folder,
                ec.experiment, ec.expe_id, fp.dataset_file, ec.base_id,
                join(fp.reference_folder, fp.training_seeding_mask),
                join(model_folder, 'model'),
                join(fp.work_expe_folder, ec.experiment, ec.expe_id, rng_seed,
                     'model', 'hyperparameters.json')]

        # STATIC ARGUMENTS
        argv += [f'--npv={ec.npv}', f'--n_actor={ec.n_actor}', f'--min_length={20}',
                 f'--max_length={200}',  # TODO : 200
                 f'--use_gpu', f'--remove_invalid_streamlines',
                 f'--scoring_data={fp.scoring_data}']

        # CALL
        scores_base = evalueate_nonrotated_tracking(argv, fp, tracking_seeds)

        # second: run all the rotated versions
        for subject_id in ec.rotation_ids:

            # FILEPATHS AND IDS
            argv = [f'{fp.ttl_root}/ttl_validation.py', curr_expe_folder,
                    ec.experiment, ec.expe_id, fp.dataset_file, subject_id,
                    join(fp.reference_folder, f'{subject_id}_wm.nii.gz'),
                    join(model_folder, 'model'),
                    join(fp.work_expe_folder, ec.experiment, ec.expe_id, rng_seed,
                              'model', 'hyperparameters.json')]

            # STATIC ARGUMENTS
            argv += [f'--npv={ec.npv}', f'--n_actor={ec.n_actor}', f'--min_length={20}',
                     f'--max_length={200}',  # TODO : 200
                     f'--use_gpu', f'--remove_invalid_streamlines',
                     f'--scoring_data={fp.scoring_data}']

            # CALL
            scores = evalueate_rotated_tracking(argv, fp, ec, subject_id, tracking_seeds)

            result_scores += [scores]

        # DUMP TO CSV
        # results without applying any rotation (base case)
        scores_base_nobundles = {k_: v_ for k_, v_ in scores_base.items() if
                                 not k_.endswith('per_bundle')}
        base_df = pd.DataFrame(scores_base_nobundles, index=['base'])
        base_df.to_csv(join(model_folder, 'base_scores.csv'))
        base_df.describe().to_csv(join(model_folder, 'base_stats.csv'))

        # collections of scores from rotated versions
        results_df = pd.DataFrame.from_records(result_scores,
                                               index=[str(idx) for idx in
                                                      range(len(result_scores))])

        results_df.to_csv(join(model_folder, 'rotated_scores.csv'))
        results_df.describe().to_csv(join(model_folder, 'scores_stats.csv'))
        results_df.describe().to_latex(join(model_folder, 'scores_stats.tex'),
                                       float_format="%.3f")


if __name__ == '__main__':
    fp_ = FilePaths()
    ec_ = ExperimentConfig()

    run_experiment(fp_, ec_)
