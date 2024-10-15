import nibabel as nib
import numpy as np
from TrackToLearn.environments.env import BaseEnv


def seeds_from_maskfile(mask_path, npv=1, rng_seed=42):
    mask = nib.load(mask_path).get_fdata().round().astype(np.uint8)
    rng = np.random.RandomState(seed=rng_seed)

    seeds = BaseEnv._get_tracking_seeds_from_mask(mask, npv, rng)

    return seeds
