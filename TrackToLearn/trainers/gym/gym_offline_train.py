#!/usr/bin/env python
import d4rl

import numpy as np
import random
import os
import torch

from collections import defaultdict
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tqdm import tqdm

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.datasets.GymDataset import GymDataset
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.trainers.gym.gym_exp import GymExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class GymOfflineTraining(GymExperiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        env_name: str,
        # RL params
        max_ep: int,
        log_interval: int,
        action_std: float,
        lr: float,
        gamma: float,
        # Model params
        hidden_dims: str,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        render: bool,
    ):
        """
        Parameters
        ----------
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        log_interval: int
            Interval at which a valid run is done
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        render: bool
            Render tracking
        """
        self.experiment_path = path
        self.experiment = experiment
        self.name = name
        self.env_name = env_name

        # RL parameters
        self.max_ep = max_ep
        self.log_interval = log_interval
        self.action_std = action_std
        self.lr = lr
        self.gamma = gamma

        #  Tracking parameters
        self.rng_seed = rng_seed

        # Model parameters
        self.use_gpu = use_gpu
        self.hidden_dims = hidden_dims
        self.render = render
        self.last_episode = 0
        self.interface_seeding = True

        # RNG
        torch.manual_seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        self.rng = np.random.RandomState(seed=self.rng_seed)
        random.seed(self.rng_seed)

        directory = pjoin(self.experiment_path, 'model')
        if not os.path.exists(directory):
            os.makedirs(directory)

    def valid(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        save_model: bool = True,
    ) -> float:
        """
        Run the tracking algorithm without noise to see how it performs

        Parameters
        ----------
        alg: RLAlgorithm
            Tracking algorithm that contains the being-trained policy
        env: BaseEnv
            Forward environment
        save_model: bool
            Save the model or not

        Returns:
        --------
        reward: float
            Reward obtained during validation
        """

        # Save the model so it can be loaded by the tracking
        if save_model:

            directory = pjoin(self.experiment_path, "model")
            if not os.path.exists(directory):
                os.makedirs(directory)
            alg.policy.save(directory, "last_model_state")

        # Launch the tracking
        reward = alg.gym_validation(
           env)

        return reward

    def rl_train(
        self,
        alg: RLAlgorithm,
        dataset,
        env: BaseEnv,
    ):
        """ Train the RL algorithm for N epochs. An epoch here corresponds to
        running tracking on the training set until all streamlines are done.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
            env: BaseEnv
                The tracking environment
            back_env: BaseEnv
                The backward tracking environment. Should be more or less
                the same as the "forward" tracking environment but initalized
                with half-streamlines
        """
        train_loader = DataLoader(dataset, shuffle=True,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size,
                                  pin_memory=True)

        # Tractogram containing all the episodes. Might be useful I guess
        # Run the valid before training to see what an untrained network does
        valid_reward = self.valid(
            alg, env)

        # Display the results of the untrained network
        self.display(env, valid_reward, 0)

        # Current epoch

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
                    states.to(device, non_blocking=True, dtype=torch.float32), \
                    actions.to(device, non_blocking=True, dtype=torch.float32), \
                    rewards.to(device, non_blocking=True, dtype=torch.float32), \
                    next_states.to(device, non_blocking=True, dtype=torch.float32), \
                    dones.to(device, non_blocking=True, dtype=torch.float32)

                losses = alg.update(
                    states, actions, rewards,
                    next_states, dones)

                add_to_means(means, losses)

                if i >= 1000:
                    break

            losses = mean_losses(means)
            print(losses)

            valid_reward = self.valid(
                alg, env)

            # Display the results of the untrained network
            self.display(env, valid_reward, 0)

    def get_dataset(self, env):
        dataset = d4rl.qlearning_dataset(env)
        return dict(
                    observations=dataset['observations'],
                    actions=dataset['actions'],
                    next_observations=dataset['next_observations'],
                    rewards=dataset['rewards'],
                    dones=dataset['terminals'].astype(np.float32),
                )

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally
        env = self.get_envs()
        # env.render()
        # Get example state to define NN input size
        example_state = env.reset()
        # env.render()
        self.input_size = example_state.shape[1]
        self.n_trajectories = example_state.shape[0]
        self.action_size = env._inner_envs[0].action_space.shape[0]

        # The RL training algorithm
        alg = self.get_alg()

        # Save hyperparameters to differentiate experiments later
        self.save_hyperparameters()

        dataset = GymDataset(self.env_name)

        # Start training !
        self.rl_train(alg, dataset, env)

        torch.cuda.empty_cache()


def add_environment_args(parser):
    parser.add_argument('env_name', type=str,
                        help='Gym env name')
