import torch
from e3nn import o3

from TrackToLearn.algorithms.so3_model import SO3Model


# user imports

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = 3  # (2)
LOG_STD_MIN = -20  # (-20)
LOG_STD_OFFSET = 2
_EPSILON = 1e-7


class SO3Actor(SO3Model):

    def __init__(self, spheredir: str, input_irrep: str, hidden_lmax: int, hidden_depth: int,
                 add_neighborhood_vox: float, neighborhood_mode: str, n_dirs: int, neighb_from_peaks: bool = False,
                 input_dim: int = -1, action_dim: int = -1):

        super(SO3Actor, self).__init__(spheredir, input_irrep, hidden_lmax, hidden_depth, add_neighborhood_vox,
                                       neighborhood_mode, n_dirs, neighb_from_peaks, input_dim, action_dim)

        # define all the state components
        self.state_config = self._init_state()

        # model
        self.model = self._get_nn_model(self.hidden_depth,
                            self.state_config,
                            self.hidden_lmax,
                            return_type=1)

    def forward(
            self,
            state: torch.Tensor,
            graph=None,
            graph_from_state=True
            ) -> (torch.Tensor, torch.Tensor):

        # make sure that model is initialized
        assert hasattr(self, 'model')

        # reorder state into individual components
        if graph_from_state:
            fiber_dict, last_dir, rel_pos = self._reformat_state(state,
                graph_from_state=graph_from_state)
            graph = self._get_graph_from_state(rel_pos, state.shape[0])
        else:
            fiber_dict, last_dir = self._reformat_state(state)

        # TODO: [x] return also the relative position from the state
        # TODO: [x] pass state_has_graph to reformat state
        # TODO: [x] greate graph based on this neighbourhood
        # TODO: [] verify.
        # TODO: [] also add this stuff to critic!

        # construct graphs from neighbourhood
        if graph is None:
            graph = self._get_graph(self.neighb_dirs, state.shape[0], dirs=last_dir)

        out = self.model(graph, fiber_dict, {})

        return out[:, 0]

class SO3DoubleCritic(SO3Model):
    def __init__(self, spheredir: str, input_irrep: str, hidden_lmax: int, hidden_depth: int,
                 add_neighborhood_vox: float, neighborhood_mode: str, n_dirs: int, neighb_from_peaks: bool = False,
                 input_dim: int = -1, action_dim: int = -1):

        super(SO3DoubleCritic, self).__init__(spheredir, input_irrep, hidden_lmax, hidden_depth, add_neighborhood_vox,
                                              neighborhood_mode, n_dirs, neighb_from_peaks, input_dim, action_dim)

        # define all the state components
        self.state_config = self._init_state(add_action=True)

        self.q1 = self._get_nn_model(self.hidden_depth,
                                     self.state_config,
                                     self.hidden_lmax,
                                     return_type=0)

        self.q2 = self._get_nn_model(self.hidden_depth,
                                     self.state_config,
                                     self.hidden_lmax,
                                     return_type=0)

    def forward(self, state, action, graph=None, graph_from_state=True) -> torch.Tensor:
        # make sure that model is initialized
        assert hasattr(self, 'q1') and hasattr(self, 'q2')

        # reorder state into individual components
        if graph_from_state:
            fiber_dict, last_dir, rel_pos = self._reformat_state(state,
                                                                 action=action,
                                                                 graph_from_state=graph_from_state)
            graph = self._get_graph_from_state(rel_pos, state.shape[0])
        else:
            fiber_dict, last_dir = self._reformat_state(state, action=action)

        # construct graphs from neighbourhood
        if graph is None:
            graph = self._get_graph(self.neighb_dirs, state.shape[0], dirs=last_dir)

        q1 = self.q1(graph, fiber_dict, {})[:, 0]
        q2 = self.q2(graph, fiber_dict, {})[:, 0]

        return q1, q2

    def Q1(self, state, action, graph=None, graph_from_state=True) -> torch.Tensor:
        assert hasattr(self, 'q1')

        # TODO: add action
        # reorder state into individual components
        if graph_from_state:
            fiber_dict, last_dir, rel_pos = self._reformat_state(state,
                                                                 action=action,
                                                                 graph_from_state=graph_from_state)
            graph = self._get_graph_from_state(rel_pos, state.shape[0])
        else:
            fiber_dict, last_dir = self._reformat_state(state, action=action)

        # construct graphs from neighbourhood
        if graph is None:
            graph = self._get_graph(self.neighb_dirs, state.shape[0], dirs=last_dir)

        # TODO: assemble

        q1 = self.q1(graph, fiber_dict, {})[:, 0]

        return q1


# noinspection PyMethodOverriding
class SO3EntropyActor(SO3Actor):

    def forward(
            self,
            state: torch.Tensor,
            stochastic: bool,
            with_logprob: bool = False,
    ) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


'''
        p = self._reformat_state(state)

        # ! below not tested! TODO implement propper with e3nn

        # log_std, mu = # TODO, extract from p

        # TODO: log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = torch.clamp(log_std + LOG_STD_OFFSET, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # pi_distribution = Normal(mu, std)
        pi_distribution = VonMisesFisher(loc=mu, scale=std)

        # stochastic=True # TODO remove

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)[..., None]
        logp_pi -= (2 * (np.log(2) - pi_action -
                         F.softplus(-2 * pi_action))).sum(axis=-1) # .sum(axis=1)

        # pi_action = self.output_activation(pi_action) # TODO

        return pi_action[:, 0], logp_pi[:, 0]
'''
