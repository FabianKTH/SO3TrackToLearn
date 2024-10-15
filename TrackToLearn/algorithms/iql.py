import copy
import numpy as np
import torch
import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple

from TrackToLearn.algorithms.sac import SAC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def add_to_means(means, dic):
    return {k: means[k] + [dic[k]] for k in dic.keys()}


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
        tanh: bool = True,
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

        self.recurrent = recurrent

        if self.tanh:
            self.output_activation = nn.Tanh()

        if self.recurrent > 0:
            rnn, decoder = make_rnn_network(
                self.hidden_layers, state_dim, action_dim, self.recurrent)
            self.rnn, self.decoder = rnn, decoder
        else:
            self.layers = make_fc_network(
                self.hidden_layers, state_dim, action_dim)

        log_std = np.zeros(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def forward(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        """

        if self.recurrent:
            return self.forward_recurrent(state, hidden)
        else:
            return self.forward_ff(state)

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

    def distribution(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):

        out, h = self.forward(state, hidden)

        std = torch.exp(self.log_std)
        mu = out

        pi_distribution = Normal(mu, std)
        return pi_distribution, h

    def act(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        stochastic: bool,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        pi_distribution, h = self.distribution(state, hidden)

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = pi_distribution.loc

        logp_pi = pi_distribution.log_prob(pi_action).sum(-1)

        if self.tanh:
            logp_pi -= (2*(np.log(2) - pi_action -
                           F.softplus(-2*pi_action))).sum(axis=1)

            pi_action = self.output_activation(pi_action)

        return pi_action, logp_pi, h

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        pi_distribution, h = self.distribution(state, hidden)

        logp_pi = pi_distribution.log_prob(action).sum(-1)

        if self.tanh:
            logp_pi -= (2*(np.log(2) - action -
                           F.softplus(-2*action))).sum(axis=1)
        return logp_pi


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function. SAC uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        recurrent: int,
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
        super(Critic, self).__init__()

        self.hidden_layers = format_widths(hidden_dims)

        self.recurrent = recurrent
        if self.recurrent > 0:
            self.q1_rnn, self.q1_decoder = make_rnn_network(
                self.hidden_layers, state_dim + action_dim, 1, self.recurrent)

            self.q2_rnn, self.q2_decoder = make_rnn_network(
                self.hidden_layers, state_dim + action_dim, 1, self.recurrent)

            self.v_rnn, self.v_decoder = make_rnn_network(
                self.hidden_layers, state_dim, 1, self.recurrent)
        else:
            self.q1 = make_fc_network(
                self.hidden_layers, state_dim + action_dim, 1)
            self.q2 = make_fc_network(
                self.hidden_layers, state_dim + action_dim, 1)
            self.v = make_fc_network(
                self.hidden_layers, state_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        """

        if self.recurrent:
            return self.forward_recurrent(state, action)
        else:
            return self.forward_ff(state, action)

    def forward_recurrent(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        padded_state, s_l = pad_packed_sequence(state, batch_first=True)
        padded_action, a_l = pad_packed_sequence(action, batch_first=True)

        padded_q1_input = torch.cat([padded_state, padded_action], -1)
        padded_q2_input = torch.cat([padded_state, padded_action], -1)

        q1_input = pack_padded_sequence(
            padded_q1_input, s_l, batch_first=True, enforce_sorted=False)
        q2_input = pack_padded_sequence(
            padded_q2_input, s_l, batch_first=True, enforce_sorted=False)

        z_q1, h_q1 = self.q1_rnn(q1_input)
        z_q2, h_q2 = self.q2_rnn(q2_input)

        z_q1 = self.q1_decoder(z_q1.data)
        z_q2 = self.q2_decoder(z_q2.data)
        if len(z_q1.shape) > 2:
            z_q1 = z_q1.squeeze(1)
            z_q2 = z_q2.squeeze(1)
        return z_q1.squeeze(-1), z_q2.squeeze(-1)

    def forward_ff(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        q1_input = torch.cat([state, action], -1)
        q2_input = torch.cat([state, action], -1)

        z_q1 = self.q1(q1_input)
        z_q2 = self.q2(q2_input)
        if len(z_q1.shape) > 2:
            z_q1 = z_q1.squeeze(1)
            z_q2 = z_q2.squeeze(1)

        return z_q1.squeeze(-1), z_q2.squeeze(-1)

    def V(self, state) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """
        if self.recurrent:

            z_v, h_v = self.v_rnn(state)
            z_v = self.v_decoder(z_v.data)
            if len(z_v.shape) > 2:
                z_v = z_v.squeeze(1)
            return z_v.squeeze(-1)
        else:

            z_v = self.v(state)
            if len(z_v.shape) > 2:
                z_v = z_v.squeeze(1)

            return z_v.squeeze(-1)


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
        recurrent: int = 0,
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
            state_dim, action_dim, hidden_dims, recurrent
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dims, recurrent
        ).to(device)

    def act(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
        stochastic: bool = True,
    ) -> torch.Tensor:
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
        a, logprob, h = self.actor.act(state, hidden, stochastic)
        return a, h, logprob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        logprob = self.actor.log_prob(state, action, hidden)
        return logprob

    def select_action(
        self,
        state: np.array,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        stochastic=False
    ) -> np.ndarray:
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
        action, h, _ = self.act(state, hidden, stochastic)

        return action.cpu().data.numpy(), h

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.actor.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict(), self.critic.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        self.critic.load_state_dict(
            torch.load(pjoin(path, filename + '_critic.pth')))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth')))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()


class IQL(SAC):
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
        gamma: float = 0.99,
        expectile: float = 0.5,
        beta: float = 1.0,
        batch_size: int = 2048,
        interface_seeding: bool = False,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda",
    ):
        """
        """

        super(IQL, self).__init__(
            input_size,
            action_size,
            hidden_dims,
            lr,
            gamma,
            beta,
            batch_size,
            interface_seeding,
            rng,
            device,
        )

        # Initialize main policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_dims, recurrent=recurrent,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        self.expectile = expectile
        self.beta = beta

        # Have to re-instanciate the optimizers for this specific
        # policy
        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # SAC-specific parameters
        self.on_policy = False

        self.start_timesteps = np.inf
        self.total_it = 0
        self.tau = 5e-3

        self.rng = rng

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        """

        reward = reward.data.squeeze(-1)
        done = done.data.squeeze(-1)

        self.policy.train()
        self.target.train()

        self.total_it += 1

        # Eq. 6
        q1_t, q2_t = self.policy.critic(state, action)
        v_tp = self.policy.critic.V(next_state).detach()
        q_target = (reward + (1. - done) * self.gamma * v_tp).detach()
        q1_loss = F.mse_loss(q1_t, q_target).mean()
        q2_loss = F.mse_loss(q2_t, q_target).mean()
        q_loss = q1_loss + q2_loss

        # Eq. 5
        q_min_t = torch.min(*self.target.critic(state, action)).detach()
        v_t = self.policy.critic.V(state)
        v_error = q_min_t - v_t
        v_sign = (v_error < 0.0).float()
        v_weight = (self.expectile - v_sign).abs().detach()
        # Assymetic squared loss
        v_loss = (v_weight * (v_error ** 2)).mean()

        critic_loss = q_loss + v_loss

        # Eq 7
        adv = q_min_t - v_t
        exp_adv = torch.exp(self.beta * adv)
        weights = torch.clamp(exp_adv, max=100.).detach()

        logp_pi = self.policy.evaluate(state, action.data)
        policy_loss = (-logp_pi * weights).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
            self.policy.critic.parameters(),
            self.target.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.policy.actor.parameters(),
            self.target.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {'policy_loss': policy_loss.detach(),
                'q1_loss': q1_loss.detach(),
                'q2_loss': q2_loss.detach(),
                'v_loss': v_loss.detach(),
                'log_pi': logp_pi.mean().detach(),
                'q1_mean': q1_t.mean().detach(),
                'q2_mean': q2_t.mean().detach(),
                'v_mean': v_t.mean().detach(),
                'adv': adv.mean().detach(),
                'reward': reward.mean(),
                }
