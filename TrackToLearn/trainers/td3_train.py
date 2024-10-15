#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.experiment.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.experiment.train import (
    add_rl_args,
    TrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class TD3TrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        td3_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        td3_train_dto: dict
            TD3 training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            td3_train_dto,
            comet_experiment,
        )

        self.seeds_provided = False # not used in training

        # TD3-specific parameters
        self.action_std = td3_train_dto['action_std']

    def save_hyperparameters(self):
        """ Add TD3-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'TD3',
             'action_std': self.action_std})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        actor_kwargs = {'spheredir': self.so3_spheredir,
                        'input_irrep': self.so3_actor_input_irrep,
                        'neighborhood_mode': self.neighborhood_mode,
                        'n_dirs': self.n_dirs,
                        'neighb_from_peaks': self.so3_neighb_from_peaks
                        }
        critic_kwargs = {}

        alg = TD3(
            actor_kwargs,
            critic_kwargs,
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_td3_args(parser):
    parser.add_argument('--action_std', default=0.3, type=float,
                        help='Action STD')

def add_so3_args(parser):
    # actor model
    parser.add_argument('--so3_actor_input_irrep',
                        default='1x0e + 1x2e + 1x4e',
                        help='irrep of input signal')
    parser.add_argument('--so3_neighb_from_peaks', action='store_true',
                        help='aligns local neighbourhood sampling to peaks')

def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_data_args(parser)
    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)
    add_td3_args(parser)
    add_so3_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace='fabiankth_ttl', parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    td3_experiment = TD3TrackToLearnTraining(
        vars(args),
        experiment
    )
    td3_experiment.run()


if __name__ == '__main__':
    main()
