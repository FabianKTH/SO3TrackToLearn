import numpy as np
import torch
from e3nn.o3 import Irreps

from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.utils.rotation import EquivarianceChecker
from TrackToLearn.utils.torch_shorthands import tnp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLAlgorithm(object):
    """
    Abstract sample-gathering and training algorithm.
    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 10000,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_size: int
            Width of the NN
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        batch_size: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        self.max_action = 1.
        self.t = 1

        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size

        self.rng = rng

    def validation_episode(
        self,
        initial_state,
        env: BaseEnv,
        compress=False,
        skip_equiv_test=False
    ):
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        tractogram: Tractogram
            Tractogram containing the tracked streamline
        running_reward: float
            Cummulative training steps reward
        """

        running_reward = 0
        state = initial_state
        done = False

        # [fabi:equiv]
        if type(self).__name__.startswith('SO3'):
            equiv = EquivarianceChecker(
                self.policy.actor.input_irrep + Irreps('1x0e'),
                self.policy.actor.state_config['n_neigh'],
                self.policy.actor.n_dirs,
                use_graph=False, # self.policy.actor.rotate_equiv_graph,
                graph_from_state=self.policy.actor.rotate_graphs)

        elif type(self).__name__.startswith('TD3'):
            # TODO initialize equivariance tester with some defaults, better:
            #  read from args
            if self.actor_kwargs['neighborhood_mode'] == 'star':
                n_neigh = 7
            else:
                raise ValueError()

            equiv = EquivarianceChecker(
                Irreps(self.actor_kwargs['input_irrep']) + Irreps('1x0e'),
                n_neigh, self.actor_kwargs['n_dirs'],
                graph_from_state=self.actor_kwargs['neighb_from_peaks'])

        else:
            skip_equiv_test = True

        equiv_hist = []

        while not np.all(done):
            # Select action according to policy + noise to make tracking
            # probabilistic

            # [fabi:equiv]
            if not skip_equiv_test:
                # TODO: sammple randomly
                n_states = 32
                perm = torch.randperm(state.shape[0])
                idx = perm[:n_states]
                rnd_state_subset = state[idx]

                equiv_hist += [tnp(equiv(rnd_state_subset, self.policy.actor))]

            action = self.policy.select_action(state)
            # Perform action
            next_state, reward, done, *_ = env.step(action)

            # Keep track of reward
            running_reward += sum(reward)

            # "Harvesting" here means removing "done" trajectories
            # from state. This line also set the next_state as the
            # state
            state, _ = env.harvest(next_state)

        if not skip_equiv_test:
            running_equiv = np.mean(equiv_hist)
            # print(f'\tEQUIV - validation: {running_equiv}')
            return running_reward, running_equiv
        else:
            return running_reward
