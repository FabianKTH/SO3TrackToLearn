import sys
import numpy as np
from distutils.dir_util import copy_tree
from os import makedirs, path
from os.path import join
from dataclasses import dataclass
from nibabel.streamlines import Tractogram
import nibabel as nib
from dipy.io.utils import get_reference_info, create_tractogram_header
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs

from TrackToLearn.searchers import so3_td3_searcher
import ttl_validation


@dataclass
class FilePaths:
    dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    ttl_root = '/home/fabi/remote_TractoRL'
    test_subject_id = 'ismrm2015'
    expe_folder = join(dset_folder, 'fabi_tests', 'experiments')
    scoring_data = '/fabi_project/data/ttl_anat_priors/fabi_tests/ismrm2015/scoring_data'
    dataset_file = join(dset_folder, 'ismrm2015',
                             test_subject_id + '_lmax4-e3nn.hdf5')
    reference_folder = join(dset_folder,  'ismrm2015/masks')

fp = FilePaths()
experiment = 'Exp3'
expe_id = "ismrm_lmax4"
no_rots = 12
# add original (no rotations)
subject_ids = [f'{fp.test_subject_id}']

# seeds=(1111 2222 3333 4444 5555)
seeds = ['1111']

for rng_seed in seeds:
    argv = list()
    model_folder = join('/fabi_project/models/Exp3/exp1-td3-fodf4', rng_seed)
    # rotation_parameters = np.load(fp.reference_folder.replace(
    #     'masks', 'rotparams.npy'),
    #     allow_pickle=True).item()
    rotation_parameters = {fp.test_subject_id: {'shift': np.array([54., 54., 54]),
                                                 'rmat': np.eye(3)}}


    result_scores = []

    for subject_id in subject_ids:
        reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')

        # FILEPATHS AND IDS
        argv = [f'{fp.ttl_root}/ttl_validation.py',
                model_folder,
                experiment,
                expe_id,
                fp.dataset_file,
                subject_id,
                join(fp.reference_folder,  f'{subject_id}_wm.nii.gz'),
                join(model_folder, 'model'),
                join(model_folder, 'model', 'hyperparameters.json')
                ]

        # STATIC ARGUMENTS
        argv += [f'--npv={1}',
                 f'--n_actor={8000}',
                 f'--min_length={20}',
                 f'--max_length={200}',
                 f'--use_gpu',
                 f'--remove_invalid_streamlines',
                 f'--scoring_data={fp.scoring_data}']

        sys.argv = argv  # hack, replace sys.argv to make the follow script accept the args

        # CALL
        tracto, exp = ttl_validation.main(return_tracto=True)
        tracto_nonlazy = Tractogram(streamlines=tracto.streamlines)
        tracto_nonlazy.affine_to_rasmm = tracto.affine_to_rasmm
        # exp.score_tractogram(tracto_nonlazy)

        scores = exp.score_tractogram(tracto_nonlazy)

        print(scores)

        result_scores += [scores]

    # dump to csv
    import pandas as pd
    results_df = pd.DataFrame.from_records(result_scores,
        index=[str(idx) for idx in range(len(result_scores))])

    results_df.to_csv(join(model_folder, 'rotated_scores.csv'))
    results_df.describe().to_csv(join(model_folder, 'scores_stats.csv'))
    results_df.describe().to_latex(join(model_folder, 'scores_stats.tex'), float_format="%.3f")
