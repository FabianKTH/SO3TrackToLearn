import numpy as np
import torch
import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple

from TrackToLearn.algorithms.utils import (
    format_widths, make_fc_network)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        tanh: bool = False,
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
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.tanh = tanh

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim * 2)

        if self.tanh:
            self.output_activation = nn.Tanh()

    def forward(
        self,
        state: torch.Tensor,
        stochastic: bool,
        with_logprob: bool = False,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        out = self.layers(state)
        mu = out[..., :self.action_dim]
        log_std = out[..., self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # log_std_logits = torch.sigmoid(log_std)
        std = torch.exp(log_std)
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

        return pi_action, logp_pi

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        out = self.layers(state)
        mu = out[..., :self.action_dim]
        log_std = out[..., self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # log_std_logits = torch.sigmoid(log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)

        logp_pi = pi_distribution.log_prob(action).sum(-1)

        if self.tanh:
            logp_pi -= (2*(np.log(2) - action -
                           F.softplus(-2*action))).sum(axis=1)
        return logp_pi


class Policy(object):
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


class GaussianActor(Actor):
    """ Actor module that takes in a state and outputs a Normal distribution.
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
        super(GaussianActor, self).__init__(
            state_dim,
            action_dim * 2,
            hidden_dims,
            recurrent,
            dropout
        )
        self.action_dim = action_dim

    def forward(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        stochastic: bool = True,
    ) -> torch.Tensor:
        """
        """

        mu_std, h = super().forward(state, hidden)
        mu = mu_std[:, :self.action_dim]
        mu = torch.tanh(mu)
        log_std = mu_std[:, self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std_logits = torch.sigmoid(log_std)
        std = torch.exp(log_std_logits)
        pi_distribution = Normal(mu, std)

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu
        logp_pi = pi_distribution.log_prob(pi_action).sum(-1)

        return pi_action, logp_pi, h

    def logprob(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:

        mu_std, h = super().forward(state, hidden)
        mu = mu_std[:, :self.action_dim]
        mu = torch.tanh(mu)
        log_std = mu_std[:, self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std_logits = torch.sigmoid(log_std)
        std = torch.exp(log_std_logits)
        pi_distribution = Normal(mu, std)

        logp_pi = pi_distribution.log_prob(action).sum(-1)

        return logp_pi


class GaussianPolicy(Policy):
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
        super(GaussianPolicy, self).__init__(
            state_dim,
            action_dim,
            hidden_dims,
            recurrent,
            dropout
        )

        self.actor = GaussianActor(
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
        a, logprob, h = self.actor(state, hidden)
        return a, logprob, h

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:

        logprob = self.actor.logprob(state, hidden, action)

        return logprob

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
        action, logprob, h = self.act(state, hidden, stochastic)

        return action.cpu().data.numpy(), h
