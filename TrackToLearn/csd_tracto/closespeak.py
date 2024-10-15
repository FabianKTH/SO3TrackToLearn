from dipy.direction import BootDirectionGetter, ClosestPeakDirectionGetter
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.direction import Sphere
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion, BinaryStoppingCriterion
from dipy.io.utils import get_reference_info, create_tractogram_header


import numpy as np
import nibabel as nib

import TrackToLearn.algorithms.so3_helper as soh
import TrackToLearn.utils.torch_shorthands
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.utils.rotation import get_ico_points

from nibabel.streamlines import Tractogram


class ClosestPeakModel:
    def __init__(self, input_data, mask_wm, mask_seeds, affine, npv, step_size, ref_file):
        # see for voxel density https://github.com/dipy/dipy/blob/master/dipy/tracking/utils.py
        self.seed_density = 1 # int(np.sqrt(npv))  # TODO 1 just for testing
        self.seeds = utils.seeds_from_mask(mask_seeds, affine, self.seed_density)
        self.stopping_criterion = BinaryStoppingCriterion(mask_wm == 1)
        self.sphere = Sphere(xyz=get_ico_points())
        self.step_size = step_size

        self.input = self.process_input(input_data)
        self.affine = affine
        self.ref_file = ref_file


    def run(self):
        peak_dg = ClosestPeakDirectionGetter.from_pmf(self.input,
                                                      max_angle=30.,
                                                      sphere=self.sphere)

        self.peak_streamline_generator = LocalTracking(peak_dg,
                                                  self.stopping_criterion,
                                                  self.seeds,
                                                  self.affine,
                                                  step_size=self.step_size)

        streamlines = Streamlines(self.peak_streamline_generator)
        tractogram = Tractogram(streamlines=streamlines, affine_to_rasmm=self.affine)

        # filename etc
        filename = '/fabi_project/TRACTO_DIPY_TEST.trk'

        filetype = nib.streamlines.detect_format(filename)
        reference = get_reference_info(self.ref_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, filename, header=header)

        
        pass
        # sft = StatefulTractogram(streamlines, self.input, Space.RASMM)

    def process_input(self, x):
        # spharm convert
        B_e3nn, invB_e3nn = soh.e3nn_sh_to_sf_matrix(
            TrackToLearn.utils.torch_shorthands.tt(self.sphere.vertices),
            lmax=4,
            basis_type='symmetric'
            )

        y = np.matmul(x, B_e3nn)

        # remove negative values TODO: is this legal?
        y[y < 0.] = 0.

        return y


class ClosestPeakExperiment:
    def __init__(self, 
                 valid_dto):
        # attach all args to class instance
        for k, v in valid_dto.items():
            setattr(self, k, v)
        
        self.load_data()

        self.model = ClosestPeakModel(
            input_data=self.input_data,
            mask_wm=self.mask_wm,
            mask_seeds=self.mask_seeds,
            affine=self.affine,
            npv=self.npv,
            step_size=self.step_size,
            ref_file=self.ref_file
            )

    def load_data(self):
        (input_volume,
         tracking_mask,
         include_mask,
         exclude_mask,
         target_mask,
         seeding_mask,
         peaks) = BaseEnv._load_dataset(self.dataset_file,
                                        self.split,
                                        self.subject_id,
                                        self.interface_seeding)

        self.input_data = input_volume.data[..., :15]
        self.mask_wm = tracking_mask.data
        self.mask_seeds = seeding_mask.data
        self.affine = input_volume.affine_vox2rasmm

    def run(self):
        self.model.run()



        
        
        
    










