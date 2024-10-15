import numpy as np
import torch
import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple

# from TrackToLearn.algorithms.shared.bc import GaussianPolicy
from TrackToLearn.algorithms.supervised import SupervisedAlgorithm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def format_widths(widths_str):
    return [int(i) for i in widths_str.split('-')]


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity, dropout=0.0
):
    layers = [nn.Linear(input_size, widths[0]), activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation(),
             nn.Dropout(dropout)])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size), last_activation()])
    return nn.Sequential(*layers)


def make_rnn_network(
    widths, input_size, output_size, n_recurrent, activation=nn.ReLU,
    last_activation=nn.Identity, dropout=0.0
):
    rnn = nn.LSTM(input_size, widths[0], num_layers=n_recurrent,
                  dropout=dropout, batch_first=True)
    layers = [activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation(),
             nn.Dropout(dropout)])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size), last_activation()])

    # no activ. on last layer
    decoder = nn.Sequential(*layers)
    return rnn, decoder


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        recurrent: int,
        dropout: float,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: int
                Widths of layers. Expected format 'width-width-[...]'

        """
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.output_activation = nn.Tanh()

        self.recurrent = recurrent
        self.tanh = False

        if self.recurrent > 0:
            rnn, decoder = make_rnn_network(
                self.hidden_layers, state_dim, action_dim, self.recurrent,
                dropout=dropout)
            self.rnn, self.decoder = rnn, decoder
        else:
            self.layers = make_fc_network(
                self.hidden_layers, state_dim, action_dim,
                dropout=dropout)

        log_std = np.zeros(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def reset(self, batch_size):
        """ Batch-first hidden state init.
        """

        if not self.recurrent:
            return None

        hidden = torch.zeros(
            (len(self.hidden_layers)-1, batch_size, self.hidden_layers[0]),
            requires_grad=True, device=device)
        cell = torch.zeros(
            (len(self.hidden_layers)-1, batch_size, self.hidden_layers[0]),
            requires_grad=True, device=device)

        return (hidden, cell)

    def forward(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        stochastic: bool = False
    ) -> torch.Tensor:
        """
        """

        if self.recurrent:
            out, h = self.forward_recurrent(state, hidden)
        else:
            out, h = self.forward_ff(state)

        std = torch.exp(self.log_std)
        mu = out
        # log_std = out[..., self.action_dim:]

        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # log_std_logits = torch.sigmoid(log_std)
        # std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu
        logp_pi = pi_distribution.log_prob(pi_action).sum(-1)

        if self.tanh:
            logp_pi -= (2*(np.log(2) - pi_action -
                           F.softplus(-2*pi_action))).sum(axis=1)
            pi_action = self.output_activation(pi_action)

        return pi_action, h

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        if self.recurrent:
            out, h = self.forward_recurrent(state, hidden)
        else:
            out, h = self.forward_ff(state)

        std = torch.exp(self.log_std)
        mu = out
        # log_std = out[..., self.action_dim:]

        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # log_std_logits = torch.sigmoid(log_std)
        # std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)

        logp_pi = pi_distribution.log_prob(action).sum(-1)

        if self.tanh:
            logp_pi -= (2*(np.log(2) - action -
                           F.softplus(-2*action))).sum(axis=1)
        return logp_pi

    def forward_recurrent(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        z, h = self.rnn(state, hidden)
        z = self.decoder(z.data)
        if len(z.shape) > 2:
            z = z.squeeze(1)
        return z, h

    def forward_ff(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        p = self.layers(state)
        if len(p.shape) > 2:
            p = p.squeeze(1)

        return p, None


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        recurrent: bool,
        dropout: float,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        self.recurrent = recurrent
        self.actor = Actor(
            state_dim, action_dim, hidden_dims, recurrent, dropout
        ).to(device)

    def act(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
        """
        # if hidden is None:
        #     hidden = self.actor.reset(state.shape[0])
        a, h = self.actor(state, hidden)
        return a, h

    def select_action(
        self, state: np.array,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
        stochastic=False
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            deterministic: bool
                Return deterministic action (at valid time)

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]
        if len(state.shape) < 3:
            state = state[:, None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action, h = self.act(state, hidden, stochastic)

        return action.cpu().data.numpy(), h

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.actor.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor
        """
        actor_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors
                will be appended
        """
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors
                will be appended
        """
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth')))

    def eval(self):
        """ Switch actors to eval mode
        """
        self.actor.eval()

    def train(self):
        """ Switch actors to train mode
        """
        self.actor.train()


class BehaviorCloning(SupervisedAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on
    TODO: Cite
    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_dims: str = '1024-1024',
        recurrent: int = 0,
        lr: float = 3e-4,
        dropout: float = 0.5,
        batch_size: int = 256,
        interface_seeding: bool = False,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda",
    ):
        """
        """

        super(BehaviorCloning, self).__init__(
            input_size,
            action_size,
            hidden_dims,
            recurrent,
            lr,
            batch_size,
            interface_seeding,
            rng,
            device,
        )

        # Initialize main policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_dims,
            recurrent=recurrent, dropout=dropout
        )

        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # SAC-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.rng = rng

    def update(
        self,
        state,
        action,
    ):
        """
        """
        self.policy.train()

        self.total_it += 1

        running_actor_loss = 0

        # pi, logp_pi, h = self.policy.act(state)
        target = action.data

        batch_logp_pi = self.policy.actor.log_prob(state, target)
        actor_loss = -batch_logp_pi.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        running_actor_loss = actor_loss.detach()

        return {'policy_loss': running_actor_loss}
