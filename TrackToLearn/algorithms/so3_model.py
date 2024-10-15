import dgl
import torch
from e3nn import o3
from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber
from torch import nn

from TrackToLearn.utils import rotation as r_util
from TrackToLearn.environments.utils import get_neighborhood_directions
from TrackToLearn.utils.torch_shorthands import tt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SO3Model(nn.Module):

    def __init__(self, spheredir: str, input_irrep: str, hidden_lmax: int, hidden_depth: int,
                 add_neighborhood_vox: float, neighborhood_mode: str, n_dirs: int, neighb_from_peaks: bool = False,
                 input_dim: int = -1, action_dim: int = -1):

        super(SO3Model, self).__init__()

        # so3 e3nn params
        self.spheredir = spheredir
        self.input_irrep = o3.Irreps(input_irrep)
        self.hidden_lmax = hidden_lmax
        self.hidden_depth = hidden_depth
        self.n_dirs = n_dirs

        # neighbourhood directions
        self.neighb_dirs = tt(get_neighborhood_directions(radius=add_neighborhood_vox,
                                                          mode=neighborhood_mode))

        # flag if graph should be aligned to the last action
        self.rotate_graphs = neighb_from_peaks

    def _init_state(self, add_action=False):
        # fodf
        self.mask_irreps = o3.Irreps('1x0e')  # (mask = interpolated binary)
        self.dirs_irreps = o3.Irreps(f'{self.n_dirs}x1o')  # (previous directions)

        if not add_action:
            self.full_input_irreps = self.input_irrep + self.mask_irreps + self.dirs_irreps
        else:
            self.full_input_irreps = self.input_irrep + self.mask_irreps + self.dirs_irreps + o3.Irreps('1x1o')

        state_config = {'fiber_sh': Fiber(self._irrep_to_fiberconf(self.input_irrep)),
                        'fiber_mask': Fiber(self._irrep_to_fiberconf(self.mask_irreps)),
                        'fiber_dirs': Fiber(self._irrep_to_fiberconf(self.dirs_irreps)),
                        'signal_dim': self.input_irrep.dim, 'n_dirs': self.n_dirs,
                        'n_neigh': self.neighb_dirs.shape[0], 'mask_dim': 1, 'dirs_dim': 3,
                        'fiber_in': Fiber(self._irrep_to_fiberconf(self.full_input_irreps)),
                        'add_action': add_action}

        return state_config

    @staticmethod
    def _irrep_to_fiberconf(irreps):
        irreps = irreps.sort().irreps.simplify()
        fconf = [(ir.l, mul) for mul, ir in irreps]

        return fconf

    @staticmethod
    def _get_nn_model(hidden_depth,
                      state_config,
                      hidden_lmax,
                      return_type
                      ):

        model = SE3Transformer(
            num_layers=hidden_depth,
            fiber_in=state_config['fiber_in'],
            fiber_hidden=Fiber.create(hidden_lmax, 8),  # 16
            fiber_out=Fiber.create(return_type + 1, 1),
            fiber_edge=Fiber({}),
            num_heads=2,  #4, 8
            channels_div=2,
            pooling='avg',  # None
            norm=True,
            use_layer_norm=True,
            # low_memory=True,
            return_type=return_type
        )

        return model

    @staticmethod
    def _get_graph(neigh, batchsize, dirs=None):
        no_nodes = neigh.shape[0]
        no_edges = no_nodes - 1

        u, v = (torch.zeros(no_edges, dtype=torch.int, device=device),
                torch.arange(1, no_nodes, dtype=torch.int, device=device))
        src_idx = torch.cat([u, v])
        dst_idx = torch.cat([v, u])

        graph = dgl.graph((src_idx, dst_idx))

        graph.edata['rel_pos'] = neigh[dst_idx] - neigh[src_idx]
        graph_batch = dgl.batch([graph] * batchsize)

        return graph_batch

    @staticmethod
    def _get_graph_from_state(rel_pos, batchsize):
        no_nodes = rel_pos.shape[1]
        no_edges = no_nodes - 1

        u, v = (torch.zeros(no_edges, dtype=torch.int, device=device),
                torch.arange(1, no_nodes, dtype=torch.int, device=device))
        src_idx = torch.cat([u, v])
        dst_idx = torch.cat([v, u])

        graph = dgl.graph((src_idx, dst_idx))
        graph_batch = dgl.batch([graph] * batchsize)

        src_idx_batch, dst_idx_batch = graph_batch.edges()

        rel_pos_flat = rel_pos.reshape(-1, 3)
        graph_batch.edata['rel_pos'] = (rel_pos_flat[dst_idx_batch]
                                        - rel_pos_flat[src_idx_batch])

        return graph_batch

    def _reformat_state(self, state, action=None, graph_from_state=False):
        sc = self.state_config

        if action is not None:
            assert sc['add_action']

        node_pos_dim = 3 if graph_from_state else 0

        assert state.shape[-1] == (sc['n_neigh'] * (sc['signal_dim']
                                                    + sc['mask_dim']
                                                    + node_pos_dim)
                                   + (sc['n_dirs'] * sc['dirs_dim']))

        sh_end = (sc['signal_dim']
                  + sc['mask_dim']
                  + node_pos_dim) * sc['n_neigh']

        sh_part, dir_part = state[:, :sh_end], state[:, sh_end:]

        assert dir_part.shape[-1] == sc['n_dirs'] * sc['dirs_dim']

        # broadcast previous directions to all nodes
        dir_part = torch.broadcast_to(dir_part[..., None, :],
                (-1, sc['n_neigh'], sc['n_dirs'] * sc['dirs_dim']))
        
        # sh_part
        sh_part = sh_part.reshape([-1,
                                   sc['n_neigh'],
                                   sc['signal_dim'] + sc['mask_dim'] + node_pos_dim])

        if graph_from_state:
            sh_coef, sh_mask, rel_pos = (sh_part[..., :sc['signal_dim']],
                                         sh_part[..., -(sc['mask_dim']+node_pos_dim):-node_pos_dim],
                                         sh_part[..., -node_pos_dim:])
        else:
            sh_coef, sh_mask = sh_part[..., :sc['signal_dim']], sh_part[..., -sc['mask_dim']:]

        # add action if provided
        if sc['add_action']:
            action = torch.broadcast_to(action[..., None, :],
                                        (-1, sc['n_neigh'], action.shape[-1]))
            x_ = torch.cat([sh_coef, sh_mask, dir_part, action], -1)
        else:
            x_ = torch.cat([sh_coef, sh_mask, dir_part], -1)

        # form fiber input dict
        fiber_dict = {}
        for ir, sl in zip(self.full_input_irreps, self.full_input_irreps.slices()):
            x_tmp = x_[..., sl].reshape(-1, ir.mul, ir.dim // ir.mul)

            lk = f'{ir.ir.l}'

            if lk in fiber_dict.keys():
                fiber_dict[lk] = torch.cat([fiber_dict[lk], x_tmp], 1)
            else:
                fiber_dict[lk] = x_tmp

        if graph_from_state:
            return fiber_dict, dir_part[:, -1, :3], rel_pos
        else:
            return fiber_dict, dir_part[:, -1, :3]


