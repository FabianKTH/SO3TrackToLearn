import sys
from dataclasses import dataclass
from os import path
from os.path import join

import nibabel as nib
import numpy as np
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs
from nibabel.streamlines import Tractogram

import ttl_validation
from TrackToLearn.utils.rotation import roto_transl, roto_transl_inv
from TrackToLearn.utils.seeds import seeds_from_maskfile

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
    # scoring_data = path.join(work_dset_folder, test_subject_id,
    # 'scoring_data')

    dataset_file = join(dset_folder, 'datasets-rotation', 'ismrm-lmax4-fix',
                        test_subject_id + '_fix_lmax4.hdf5')
    reference_folder = join(dset_folder, 'datasets-rotation', 'ismrm-lmax4-fix',
                            'masks')

    training_seeding_mask = ('/fabi_project/data/datasets/ismrm2015/masks'
                             '/ismrm2015_wm.nii.gz')

    so3_spheredir = '/fabi_project/sphere'


fp = FilePaths()
experiment = 'Exp3'
# ID=$(date +"%F-%H_%M_%S")
expe_id = f"exp1-so3-fix"
# seeds=(1111 2222 3333 4444 5555)

no_rots = 12
subject_ids = [f'{fp.test_subject_id}-rotation{rot_idx}' for rot_idx in
               range(no_rots)]
# add original (no rotations)
# subject_ids = [f'{fp.test_subject_id}'] + subject_ids

# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
    curr_expe_folder = join(fp.work_expe_folder, experiment, expe_id, rng_seed)

    # rotation_parameters = np.load(
    #         fp.reference_folder.replace('masks', 'rotparams.npy'),
    #         allow_pickle=True).item()

    result_scores = []
    npv = 1
    padshift = np.array([9., 0., 9.])


    tracking_seeds = seeds_from_maskfile(fp.training_seeding_mask, npv,
                                         int(rng_seed)
                                         )

    # !! TODO remove, just for quicker debugging
    # rng = np.random.default_rng(int(rng_seed))
    # tracking_seeds = rng.choice(a=tracking_seeds, size=128, replace=False,
    #                             axis=0)

    # !! TODO remove next line, testing
    # subject_ids= ['ismrm2015-rotation5']

    for subject_id in subject_ids:
        reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')
        model_folder = join(fp.work_expe_folder, experiment, expe_id, rng_seed)

        # FILEPATHS AND IDS
        argv = [f'{fp.ttl_root}/ttl_validation.py', curr_expe_folder,
                experiment, expe_id, fp.dataset_file, subject_id,
                join(fp.reference_folder, f'{subject_id}_wm.nii.gz'),
                join(model_folder, 'model'),
                path.join(fp.work_expe_folder, experiment, expe_id, rng_seed,
                          'model', 'hyperparameters.json')]

        # STATIC ARGUMENTS
        argv += [f'--npv={npv}', f'--n_actor={2 ** 8}', f'--min_length={20}',
                 f'--max_length={200}',
                 f'--use_gpu', f'--remove_invalid_streamlines',
                 f'--scoring_data={fp.scoring_data}']

        sys.argv = argv  # hack, replace sys.argv to make the follow script
        # accept the args

        # load rotation matrix
        linear_mat = np.loadtxt(join(fp.reference_folder.replace('masks', subject_id + '.txt')))
        rmat = linear_mat[:3, :3]
        shift = linear_mat[-1, :3]

        # rotate preloaded tracking seeds
        seeds_rotated = roto_transl_inv(tracking_seeds, rmat, shift,
                                    padshift=padshift)

        # continue # TODO remove!!
        # seeds_rot_sampled = seeds_from_maskfile(reference_file, npv,
        #                                          int(rng_seed))

        # CALL
        tracto_dict = ttl_validation.main(return_tracto=True,
                                          seeds_provided=True,
                                          seed_array=seeds_rotated)

        nib.streamlines.save(tracto_dict['tractogram'], tracto_dict['filename'],
                             header=tracto_dict['header'])

        basic_bundles_attribs = load_attribs(
            path.join(fp.scoring_data, 'gt_bundles_attributes.json'))

        # TODO!! rotate tractogram back in order to score submission
        # rparams = rotation_parameters[subject_id]

        streamlines = [sl.streamline for sl in tracto_dict['tractogram']]
        streamlines_rotated = [
                roto_transl((sldat/tracto_dict['vox_size']) - 0.5,
                            rmat, shift, padshift=padshift) + 0.5 for
                sldat in streamlines]

        tracto_invrot = Tractogram(streamlines=streamlines_rotated)
        tracto_invrot.affine_to_rasmm = tracto_dict['affine_vox2rasmm']
        filename_rotated = tracto_dict['filename'].replace('.trk',
                                                           '_invrot.trk')
        nib.streamlines.save(tracto_invrot, filename_rotated,
                             header=tracto_dict['header'])
        # tracto_invrot_path = join(model_folder, 'trackt_rotated',
        # 'tractogram_{}.trk'.format(subject_id))

        # scores = exp.score_tractogram(tracto_invrot)
        scores = score_submission(filename_rotated, fp.scoring_data,
                                  basic_bundles_attribs, compute_ic_ib=True)

        print(scores)

        result_scores += [scores]

    # dump to csv
    import pandas as pd

    results_df = pd.DataFrame.from_records(result_scores,
                                           index=[str(idx) for idx in
                                                  range(len(result_scores))])

    results_df.to_csv(join(model_folder, 'rotated_scores.csv'))
    results_df.describe().to_csv(join(model_folder, 'scores_stats.csv'))
    results_df.describe().to_latex(join(model_folder, 'scores_stats.tex'),
                                   float_format="%.3f")
