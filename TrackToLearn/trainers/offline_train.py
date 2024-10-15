#!/usr/bin/env python
import numpy as np
import random
import os
import torch

from collections import defaultdict
from comet_ml import Experiment
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tqdm import tqdm

from TrackToLearn.datasets.StreamlineDataset import StreamlineDataset
from TrackToLearn.datasets.utils import BucketSampler, Collater
from TrackToLearn.environments.interface_tracker import InterfaceTrackingEnvironment

from TrackToLearn.experiment.ttl import TrackToLearnExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TrackToLearnOfflineTraining(TrackToLearnExperiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        dataset_file: str,
        reference_file: str,
        scoring_data: str,
        max_ep: int,
        gradient_steps_per_epoch: int,
        lr: float,
        gamma: float,
        batch_size: int,
        n_dirs: int,
        dense_rewards: bool,
        reward_scaling: int,
        interface_seeding: bool,
        # Tracking params
        npv: int,
        theta: float,
        min_length: int,
        max_length: int,
        step_size: float,  # Step size (in mm)
        # Model params
        hidden_dims: str,
        recurrent: int,
        add_neighborhood: float,
        # Experiment params
        precomputed_states: bool,
        num_workers: int,
        use_gpu: bool,
        rng_seed: int,
        comet_experiment: Experiment,
        run_tractometer: bool,
        render: bool,
    ):
        """
        """
        self.experiment_path = path
        self.experiment = experiment
        self.name = name

        # RL parameters
        self.max_ep = max_ep
        self.gradient_steps_per_epoch = gradient_steps_per_epoch
        self.lr = lr
        self.gamma = gamma

        print(add_neighborhood)

        #  Tracking parameters
        self.add_neighborhood = add_neighborhood
        self.dataset_file = dataset_file
        self.reference_file = reference_file
        self.scoring_data = scoring_data
        self.rng_seed = rng_seed
        self.interface_seeding = interface_seeding
        self.dense_rewards = dense_rewards
        self.reward_scaling = reward_scaling
        self.npv = npv
        self.theta = theta
        self.min_length = min_length
        self.max_length = max_length
        self.step_size = step_size

        # Model parameters
        self.precomputed_states = precomputed_states
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.hidden_dims = hidden_dims
        self.recurrent = recurrent
        self.comet_experiment = comet_experiment
        self.render = render
        self.last_episode = 0
        self.batch_size = batch_size
        self.n_dirs = n_dirs

        self.compute_reward = True  # Always compute reward during training
        self.fa_map = None
        self.n_actor = 5000
        self.valid_subject_id = 'FiberCup'  # TODO: change this
        self.run_tractometer = run_tractometer

        # RNG
        torch.manual_seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        self.rng = np.random.RandomState(seed=self.rng_seed)
        random.seed(self.rng_seed)

        directory = pjoin(self.experiment_path, 'model')
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_envs(self):
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        # Not sure if parameters should come from `self` of actual
        # function parameters. It feels a bit dirty to have everything
        # in `self`, but then there's a crapload of parameters

        # Forward environment
        env = InterfaceTrackingEnvironment.from_dataset(
            self.dataset_file,
            "training",
            interface_seeding=self.interface_seeding,
            n_dirs=self.n_dirs,
            step_size=self.step_size,
            theta=self.theta,
            min_length=self.min_length,
            max_length=self.max_length,
            npv=self.npv,
            rng=self.rng,
            add_neighborhood=self.add_neighborhood,
            compute_reward=True,
            device=device)
        back_env = None

        return back_env, env

    def get_valid_envs(self):
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        # Not sure if parameters should come from `self` of actual
        # function parameters. It feels a bit dirty to have everything
        # in `self`, but then there's a crapload of parameters

        env = InterfaceTrackingEnvironment.from_dataset(
            self.dataset_file,
            "validation",
            interface_seeding=self.interface_seeding,
            n_dirs=self.n_dirs,
            step_size=self.step_size,
            theta=self.theta,
            min_length=self.min_length,
            max_length=self.max_length,
            npv=self.npv,
            rng=self.rng,
            add_neighborhood=self.add_neighborhood,
            compute_reward=True,
            device=device)

        back_env = None

        return back_env, env

    def rl_train(
        self,
        train_dataset: StreamlineDataset,
        valid_dataset: StreamlineDataset,
    ):
        """ Train the Offline RL algorithm for N epochs. An epoch here corresponds to
        going through the training dataset once.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
        """
        # The RL training algorithm
        alg = self.get_alg()

        back_env, env = self.get_envs()
        back_valid_env, valid_env = self.get_valid_envs()

        collate_fn = Collater(train_dataset, self.recurrent > 0)
        sampler = BucketSampler(train_dataset.lengths, shuffle=True,
                                batch_size=self.batch_size,
                                number_of_batches=self.gradient_steps_per_epoch)
        train_loader = DataLoader(train_dataset,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_fn,
                                  batch_sampler=sampler,
                                  # batch_size=self.batch_size,
                                  prefetch_factor=2,
                                  persistent_workers=True
                                  if self.num_workers > 0 else False,
                                  pin_memory=True)

        # valid_loader = DataLoader(
        #     valid_dataset, batch_size=self.batch_size, shuffle=True,
        #     num_workers=1, collate_fn=tracto_collate_fn)
        # test_loader = DataLoader(
        #     valid_dataset, batch_size=self.batch_size, shuffle=True,
        #     num_workers=1, collate_fn=tracto_collate_fn)

        # Validation run
        valid_tractogram, valid_reward = self.valid(
            alg, valid_env, back_valid_env)

        self.display(valid_tractogram, env, valid_reward, 0)

        def add_to_means(means, dic):
            return {k: means[k].append(dic[k]) for k in dic.keys()}

        def mean_losses(dic):
            return {k: torch.mean(torch.stack(dic[k])).cpu().numpy()
                    for k in dic.keys()}

        for epoch in range(self.max_ep):
            print("Epoch: {} of {}".format(epoch + 1, self.max_ep))
            means = defaultdict(list)

            for i, (
                states, actions, rewards, next_states, dones
            ) in enumerate(tqdm(train_loader), 0):
                # transfer tensors to selected device
                states, actions, rewards, next_states, dones = \
                    states.to(device, non_blocking=True), \
                    actions.to(device, non_blocking=True), \
                    rewards.to(device, non_blocking=True), \
                    next_states.to(device, non_blocking=True), \
                    dones.to(device, non_blocking=True)

                losses = alg.update(
                    states, actions, rewards,
                    next_states, dones)

                add_to_means(means, losses)

            # Validation run
            valid_tractogram, valid_reward = self.valid(
                alg, valid_env, back_valid_env)

            # Display what the network is capable-of "now"
            self.display(
                valid_tractogram,
                env,
                valid_reward,
                epoch + 1,
                self.run_tractometer)

            losses = mean_losses(means)
            print(losses)

            if self.comet_experiment is not None:
                self.comet_monitor.log_losses(losses, epoch)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        train_dataset = StreamlineDataset(
            self.dataset_file, 'training', self.n_dirs, self.add_neighborhood,
            dense_rewards=self.dense_rewards,
            reward_scaling=self.reward_scaling)
        valid_dataset = StreamlineDataset(
            self.dataset_file, 'validation', self.n_dirs, self.add_neighborhood,
            dense_rewards=self.dense_rewards,
            reward_scaling=self.reward_scaling)
        valid_dataset = StreamlineDataset(
            self.dataset_file, 'testing', self.n_dirs, self.add_neighborhood,
            dense_rewards=self.dense_rewards,
            reward_scaling=self.reward_scaling)

        # Get example state to define NN input size
        self.input_size = train_dataset.state_size

        # Save hyperparameters to differentiate experiments later
        self.save_hyperparameters()

        self.setup_monitors()

        # Setup comet monitors to monitor experiment as it goes along
        if self.comet_experiment:
            self.setup_comet()

        oom = True

        while oom:
            try:
                # Start training !
                self.rl_train(train_dataset, valid_dataset, valid_dataset)
                oom = False
            except RuntimeError as e:
                raise e
                print(e)
                print('Ran out of memory, splitting batch size in half')
                torch.cuda.empty_cache()
                self.batch_size = self.batch_size // 2
                print(
                    f'Batch size is now {self.batch_size}. Restarting training'
                )


def add_rl_args(parser):
    parser.add_argument('--max_ep', default=500, type=int,
                        help='Number of epochs.')
    parser.add_argument('--lr', default=1e-6, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.925, type=float,
                        help='Gamma param for reward discounting')
    parser.add_argument('--gradient_steps_per_epoch', default=250, type=int,
                        help='Number of gradient steps per epoch.')
    parser.add_argument('--dense_rewards', action='store_true',
                        help='Wheter to have sparse or dense rewards.')
    parser.add_argument('--reward_scaling', default=1.0, type=float,
                        help='Reward multiplication factor.')
