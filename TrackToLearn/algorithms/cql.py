import copy
import numpy as np
import torch
import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal

from TrackToLearn.algorithms.sac import SAC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def extend_and_repeat(tensor, repeat):
    repeated = tensor.unsqueeze(1)
    repeated = torch.repeat_interleave(repeated, repeat, dim=1)
    repeated = repeated.view(-1, *tensor.shape[1:])
    return repeated


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

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim)

        if self.tanh:
            self.output_activation = nn.Tanh()

        log_std = np.zeros(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

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
        # mu = out[..., :self.action_dim]
        # log_std = out[..., self.action_dim:]

        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # # log_std_logits = torch.sigmoid(log_std)
        # std = torch.exp(log_std)

        std = torch.exp(self.log_std)
        mu = out
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
        # mu = out[..., :self.action_dim]
        # log_std = out[..., self.action_dim:]

        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # # log_std_logits = torch.sigmoid(log_std)
        # std = torch.exp(log_std)

        std = torch.exp(self.log_std)
        mu = out
        pi_distribution = Normal(mu, std)

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

        self.q1 = make_fc_network(
            self.hidden_layers, state_dim + action_dim, 1)
        self.q2 = make_fc_network(
            self.hidden_layers, state_dim + action_dim, 1)
        self.v = make_fc_network(
            self.hidden_layers, state_dim, 1)

    def forward(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        q1_input = torch.cat([state, action], -1)
        q2_input = torch.cat([state, action], -1)

        q1 = self.q1(q1_input).squeeze(-1)
        q2 = self.q2(q2_input).squeeze(-1)

        return q1, q2

    def V(self, state) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """
        v = self.v(state).squeeze(-1)

        return v


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
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
        self.actor = Actor(
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dims
        ).to(device)

    def act(
        self,
        state: torch.Tensor,
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
        a, logprob = self.actor(state, stochastic)
        return a, logprob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ):
        logprob = self.actor.log_prob(state, action)
        return logprob

    def select_action(self, state: np.array, stochastic=False) -> np.ndarray:
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

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action, _ = self.act(state, stochastic)

        return action.cpu().data.numpy(), None

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


class CQL(SAC):
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
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        min_q_weight: float = 5.0,
        backup_entropy: bool = False,
        automatic_entropy_tuning: bool = True,
        batch_size: int = 2048,
        interface_seeding: bool = False,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda",
    ):
        """
        """

        super(CQL, self).__init__(
            input_size,
            action_size,
            hidden_dims,
            lr,
            gamma,
            alpha,
            batch_size,
            interface_seeding,
            rng,
            device,
        )

        # # Initialize main policy
        # self.policy = ActorCritic(
        #     input_size, action_size, hidden_dims,
        # )

        # # Initialize target policy to provide baseline
        # self.target = copy.deepcopy(self.policy)

        # # Have to re-instanciate the optimizers for this specific
        # # policy
        # # Optimizer for critic
        # self.critic_optimizer = torch.optim.Adam(
        #     self.policy.critic.parameters(), lr=lr)

        # # Optimizer for actor
        # self.actor_optimizer = torch.optim.Adam(
        #     self.policy.actor.parameters(), lr=lr)

        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Auto-temperature adjustment
        starting_temperature = np.log(alpha)  # Found empirically
        self.target_entropy = -np.prod(action_size).item()
        self.entropy_multiplier = 1.0
        self.log_alpha = torch.full(
            (1,), starting_temperature, requires_grad=True, device=device)

        if self.automatic_entropy_tuning:
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=lr)
        else:
            self.log_alpha.requires_grad = False

        self.min_q_weight = min_q_weight
        self.backup_entropy = backup_entropy

        # SAC-specific parameters
        self.on_policy = False

        self.start_timesteps = 0
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

        reward = reward.squeeze(-1)
        done = done.squeeze(-1)

        self.policy.train()
        self.target.train()

        self.total_it += 1

        running_actor_loss = 0
        running_critic_loss = 0

        pi, logp_pi = self.policy.act(state)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                logp_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp() * self.entropy_multiplier
        else:
            alpha = torch.exp(self.log_alpha)

        q_pi = torch.min(*self.policy.critic(state, pi))

        # Entropy-regularized policy loss
        if self.total_it <= self.start_timesteps:
            batch_logp_pi = self.policy.actor.logprob(state, action)
            actor_loss = (alpha.detach() * logp_pi - batch_logp_pi).mean()
        else:
            actor_loss = (alpha * logp_pi - q_pi).mean()

        # Target action come from *current* policy
        next_action, logp_next_action = self.policy.act(next_state)

        # Compute the target Q value
        target_Q1, target_Q2 = self.target.critic(
            next_state, next_action)
        backup = torch.min(target_Q1, target_Q2)

        if self.backup_entropy:
            backup = backup - alpha * logp_next_action

        backup = (reward + (1. - done) * self.gamma * backup).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.policy.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(current_Q1, backup)
        loss_q2 = F.mse_loss(current_Q2, backup)

        # CQL
        batch_size = action.shape[0]
        action_dim = action.shape[-1]

        repeated_states = extend_and_repeat(state, 10)

        repeated_next_states = extend_and_repeat(next_state, 10)

        random_action = action.new_empty(
            (batch_size * 10, action_dim),
            requires_grad=False).uniform_(-1, 1)

        current_action, current_logp_pi = self.policy.act(
            repeated_states)
        next_action, next_logp_pi = self.policy.act(
            repeated_next_states)

        current_log_pis = torch.reshape(
            current_logp_pi, (batch_size, 10)).detach()
        next_log_pis = torch.reshape(
            next_logp_pi, (batch_size, 10)).detach()

        q1_rand, q2_rand = self.policy.critic(
            extend_and_repeat(state, 10), random_action)
        q1_current_action, q2_current_action = self.policy.critic(
            extend_and_repeat(state, 10), current_action.detach())
        q1_next_action, q2_next_action = self.policy.critic(
            extend_and_repeat(state, 10), next_action.detach())

        q1_rand = torch.reshape(q1_rand, (batch_size, 10))
        q1_current_action = torch.reshape(
            q1_current_action, (batch_size, 10))
        q1_next_action = torch.reshape(
            q1_next_action, (batch_size, 10))

        q2_rand = torch.reshape(q2_rand, (batch_size, 10))
        q2_current_action = torch.reshape(
            q2_current_action, (batch_size, 10))
        q2_next_action = torch.reshape(
            q2_next_action, (batch_size, 10))

        random_density = np.log(0.5 ** action_dim)
        cat_q1 = torch.cat(
            [q1_rand - random_density,
             q1_next_action - next_log_pis.detach(),
             q1_current_action - current_log_pis.detach()],
            dim=1
        )
        cat_q2 = torch.cat(
            [q2_rand - random_density,
             q2_next_action - next_log_pis.detach(),
             q2_current_action - current_log_pis.detach()],
            dim=1
        )

        delta_k1 = (q1_rand - q1_current_action).mean().detach()
        delta_k2 = (q2_rand - q2_current_action).mean().detach()

        qf1_ood = torch.logsumexp(cat_q1, dim=1)
        qf2_ood = torch.logsumexp(cat_q2, dim=1)

        """Subtract the log likelihood of data"""
        qf1_diff = (qf1_ood - current_Q1).mean()
        qf2_diff = (qf2_ood - current_Q2).mean()

        min_qf1_loss = qf1_diff * self.min_q_weight
        min_qf2_loss = qf2_diff * self.min_q_weight

        critic_loss = loss_q1 + loss_q2 + min_qf1_loss + min_qf2_loss

        if self.automatic_entropy_tuning:
            # Optimize the temperature
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

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

        running_actor_loss = actor_loss.detach()

        running_critic_loss = critic_loss.detach()

        return {'policy_loss': running_actor_loss,
                'critic_loss': running_critic_loss,
                'q1_loss': loss_q1.detach(),
                'q2_loss': loss_q2.detach(),
                'qf1_loss': min_qf1_loss.detach(),
                'qf2_loss': min_qf2_loss.detach(),
                'delta_k1': delta_k1,
                'delta_k2': delta_k2,
                'log_pi': logp_pi.mean().detach(),
                'q1_mean': current_Q1.mean().detach(),
                'q2_mean': current_Q2.mean().detach(),
                'alpha': self.log_alpha.detach(),
                'reward': reward.mean(),
                }
