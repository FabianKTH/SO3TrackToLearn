#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import json
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment
from os.path import join as pjoin

from TrackToLearn.algorithms.behavior_cloning import BehaviorCloning
from TrackToLearn.experiment.experiment import (
    add_offline_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.trainers.supervised_train import (
    add_supervised_args,
    TrackToLearnSupervisedTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class BehaviorCloningTrackToLearnTraining(TrackToLearnSupervisedTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        path,
        experiment,
        name,
        dataset_file,
        reference_file,
        scoring_data,
        max_ep,
        gradient_steps_per_epoch,
        lr,
        dropout,
        batch_size,
        n_dirs,
        interface_seeding,
        npv,
        theta,
        min_length,
        max_length,
        step_size,
        hidden_dims,
        recurrent,
        add_neighborhood,
        precomputed_states,
        num_workers,
        use_gpu,
        rng_seed,
        comet_experiment,
        run_tractometer,
        render
    ):
        """
        Parameters
        ----------
        dataset_file: str
            Path to the file containing the signal data
        subject_id: str
            Subject being trained on (in the signal data)
        in_seed: str
            Path to the mask where seeds can be generated
        in_mask: str
            Path to the mask where tracking can happen
        scoring_data: str
            Path to reference streamlines that can be used for
            jumpstarting seeds
        target_file: str
            Path to the mask representing the tracking endpoints
        exclude_file: str
            Path to the mask reprensenting the tracking no-go zones
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        gradient_steps_per_epoch: int
            Interval at which a valid run is done
        lr: float
            Learning rate for optimizer
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
        training_batch_size: int
            How many samples to use in policy update
        npv: int
            How many seeds to generate per voxel
        theta: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        step_size: float
            Step size for tracking
        alignment_weighting: float
            Reward coefficient for alignment with local odfs peaks
        straightness_weighting: float
            Reward coefficient for streamline straightness
        length_weighting: float
            Reward coefficient for streamline length
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        n_actor: int
            Batch size for tracking during valid
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        comet_experiment: Experiment
            Use comet for displaying stats during training
        render: bool
            Render tracking (to file)
        run_tractometer: bool
            Run tractometer during validation to see how it's
            doing w.r.t. ground truth data
        load_teacher: str
            Path to pretrained model for imitation learning
        load_policy: str
            Path to pretrained policy
        """

        self.batch_size = batch_size

        super().__init__(
            # Dataset params
            path,
            experiment,
            name,
            dataset_file,
            reference_file,
            scoring_data,
            # SAC params
            max_ep,
            gradient_steps_per_epoch,
            lr,
            # Env params
            batch_size,
            n_dirs,
            interface_seeding,
            # Tracking params
            npv,
            theta,
            min_length,
            max_length,
            step_size,
            # Model params
            hidden_dims,
            recurrent,
            add_neighborhood,
            # Experiment params
            precomputed_states,
            num_workers,
            use_gpu,
            rng_seed,
            comet_experiment,
            run_tractometer,
            render,
        )

        self.action_size = 3
        self.dropout = dropout

    def save_hyperparameters(self):
        self.hyperparameters = {
            # RL parameters
            'id': self.name,
            'experiment': self.experiment,
            'algorithm': 'BC',
            'max_ep': self.max_ep,
            'gradient_steps_per_epoch': self.gradient_steps_per_epoch,
            'lr': self.lr,
            'dropout': self.dropout,
            # Data parameters
            'input_size': self.input_size,
            'add_neighborhood': self.add_neighborhood,
            'random_seed': self.rng_seed,
            'dataset_file': self.dataset_file,
            # Model parameters
            'experiment_path': self.experiment_path,
            'use_gpu': self.use_gpu,
            'hidden_dims': self.hidden_dims,
            'recurrent': self.recurrent,
            'last_episode': self.last_episode,
            'batch_size': self.batch_size,
            'n_dirs': self.n_dirs,
            'interface_seeding': self.interface_seeding,
            # Reward parameters
        }

        directory = pjoin(self.experiment_path, "model")
        with open(
            pjoin(directory, "hyperparameters.json"),
            'w'
        ) as json_file:
            json_file.write(
                json.dumps(
                    self.hyperparameters,
                    indent=4,
                    separators=(',', ': ')))

    def get_alg(self):
        alg = BehaviorCloning(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.recurrent,
            self.lr,
            self.dropout,
            self.batch_size,
            self.interface_seeding,
            self.rng,
            device)
        return alg


def add_cosine_args(parser):
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate.')
    parser.add_argument('--training_batch_size', default=2**14, type=int,
                        help='Number of seeds used per episode')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_offline_data_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_supervised_args(parser)
    add_tracking_args(parser)

    add_cosine_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    # torch.multiprocessing.set_start_method("spawn")
    experiment = Experiment(project_name=args.experiment,
                            workspace='TrackToLearn', parse_args=False,
                            auto_metric_logging=False,
                            disabled=not args.use_comet)

    behavior_cloning_experiment = BehaviorCloningTrackToLearnTraining(
        # Dataset params
        args.path,
        args.experiment,
        args.name,
        args.dataset_file,
        args.reference_file,
        args.scoring_data,
        args.max_ep,
        args.gradient_steps_per_epoch,
        args.lr,
        args.dropout,
        args.batch_size,
        args.n_dirs,
        args.interface_seeding,
        args.npv,
        args.theta,
        args.min_length,
        args.max_length,
        args.step_size,
        args.hidden_dims,
        args.recurrent,
        args.add_neighborhood,
        args.precomputed_states,
        args.num_workers,
        args.use_gpu,
        args.rng_seed,
        experiment,
        args.run_tractometer,
        args.render
    )
    behavior_cloning_experiment.run()


if __name__ == '__main__':
    main()
