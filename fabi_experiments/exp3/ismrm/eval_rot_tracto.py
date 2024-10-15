import sys
from os.path import join

import nibabel as nib
import numpy as np
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs

import ttl_validation
from TrackToLearn.utils.rotation import roto_transl_inv, roto_transl


def evalueate_nonrotated_tracking(argv, fp, tracking_seeds):
    # reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')
    # seeds_rot_sampled = seeds_from_maskfile(reference_file, npv,
    #                                          int(rng_seed))

    sys.argv = argv  # hack, replace sys.argv to make the follow script
    # accept the args
    # CALL
    tracto_dict = ttl_validation.main(return_tracto=True,
                                      seeds_provided=True,
                                      seed_array=tracking_seeds)
    nib.streamlines.save(tracto_dict['tractogram'], tracto_dict['filename'],
                         header=tracto_dict['header'])
    # scoring
    basic_bundles_attribs = load_attribs(
        join(fp.scoring_data, 'gt_bundles_attributes.json'))
    scores = score_submission(tracto_dict['filename'], fp.scoring_data,
                              basic_bundles_attribs, compute_ic_ib=True)
    print(scores)
    return scores


def evalueate_rotated_tracking(argv, fp, ec, subject_id, tracking_seeds):
    # load rotation matrix
    linear_mat = np.loadtxt(join(fp.reference_folder.replace('masks', subject_id + '.txt')))
    rmat = linear_mat[:3, :3]
    shift = linear_mat[-1, :3]
    # rotate preloaded tracking seeds
    seeds_rotated = roto_transl_inv(tracking_seeds, rmat, shift,
                                    padshift=ec.padshift)
    # reference_file = join(fp.reference_folder, subject_id + '_wm.nii.gz')
    # seeds_rot_sampled = seeds_from_maskfile(reference_file, npv,
    #                                          int(rng_seed))
    sys.argv = argv  # hack, replace sys.argv to make the follow script
    # accept the args
    # CALL
    tracto_dict = ttl_validation.main(return_tracto=True,
                                      seeds_provided=True,
                                      seed_array=seeds_rotated)
    nib.streamlines.save(tracto_dict['tractogram'], tracto_dict['filename'],
                         header=tracto_dict['header'])
    # rotate tractogram back in order to score submission
    streamlines = [sl.streamline for sl in tracto_dict['tractogram']]
    streamlines_rotated = [
        roto_transl((sldat / tracto_dict['vox_size']) - 0.5,
                    rmat, shift, padshift=ec.padshift) + 0.5 for
        sldat in streamlines]
    tracto_invrot = nib.streamlines.Tractogram(streamlines=streamlines_rotated)
    tracto_invrot.affine_to_rasmm = tracto_dict['affine_vox2rasmm']
    filename_rotated = tracto_dict['filename'].replace('.trk',
                                                       '_invrot.trk')
    nib.streamlines.save(tracto_invrot, filename_rotated,
                         header=tracto_dict['header'])
    # scoring
    basic_bundles_attribs = load_attribs(
        join(fp.scoring_data, 'gt_bundles_attributes.json'))
    scores = score_submission(filename_rotated, fp.scoring_data,
                              basic_bundles_attribs, compute_ic_ib=True)
    print(scores)
    return scores
