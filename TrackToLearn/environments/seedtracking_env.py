import numpy as np


from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.datasets.utils import MRIDataVolume

class SeedTrackingEnvironment(TrackingEnvironment):
    """ SeedTracking environment for tracking from a pre-defined set of
        seedpoints.
    """

    def __init__(
            self,
            input_volume: MRIDataVolume,
            tracking_mask: MRIDataVolume,
            target_mask: MRIDataVolume,
            seeding_mask: MRIDataVolume,
            peaks: MRIDataVolume,
            env_dto: dict,
            seeds: np.array ,
            include_mask: MRIDataVolume = None,
            exclude_mask: MRIDataVolume = None,
            ):

        super(TrackingEnvironment, self).__init__(input_volume,
                                                  tracking_mask, target_mask,
                                                  seeding_mask, peaks,
                                                  env_dto, include_mask,
                                                  exclude_mask)

        # replace seeds with the seeds provided
        self.seeds = seeds
        print(
            'After replacement: {} has {} seeds.'.format(
                    self.__class__.__name__,
                                      len(self.seeds)))

    def nreset(self, n_seeds: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines.
            Picks just the first n seeds without any randomness.
        Parameters
        ----------
        n_seeds: int
            How many seeds to sample

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        self.initial_points = self.seeds[:n_seeds]

        self.streamlines = np.zeros(
            (n_seeds, self.max_nb_steps + 1, 3), dtype=np.float32)
        self.streamlines[:, 0, :] = self.initial_points

        self.flags = np.zeros(n_seeds, dtype=int)

        self.lengths = np.ones(n_seeds, dtype=np.int32)

        self.length = 1

        # Initialize rewards and done flags
        self.dones = np.full(n_seeds, False)
        self.continue_idx = np.arange(n_seeds)

        # Setup input signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    @classmethod
    def from_dataset(
        cls,
        env_dto: dict,
        split: str,
        seeds: np.array = None
    ):
        dataset_file = env_dto['dataset_file']
        subject_id = env_dto['subject_id']
        interface_seeding = False
        assert seeds is not None

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks,
         ) = BaseEnv._load_dataset(
                dataset_file, split, subject_id, interface_seeding
        )

        return cls(
            input_volume,
            tracking_mask,
            target_mask,
            seeding_mask,
            peaks,
            env_dto,
            seeds=seeds,
            include_mask=include_mask,
            exclude_mask=exclude_mask,
        )