import os
import torch
import numpy as np
from TrackToLearn.utils.torch_shorthands import tt, tnp
from e3nn import o3
from TrackToLearn.assets import sphere


epsilon = 1e-7


def get_ico_points(subdiv=4, fname=None):

    spheredir = sphere.__path__._path[0]

    if not fname:
        fname = f'ico{subdiv}.npy'

    if fname.endswith('.vtk'):
        fname = fname.replace('.vtk', '.npy')

    numpy_array_of_points = np.load(os.path.join(spheredir, fname))

    return numpy_array_of_points


def align_a_to_b(a, b):
    """
    returns batch of rotation matrices to align batch of a
    with batch of b.
    
    adapted from https://stackoverflow.com/a/67767180
    """
    dot = lambda x, y: torch.einsum('...i,...i->...', x, y)

    bs = a.shape[0]

    a = a / torch.clamp_min(torch.norm(a, dim=-1, keepdim=True), epsilon)
    b = b / torch.clamp_min(torch.norm(b, dim=-1, keepdim=True), epsilon)
    c = dot(a, b)
    v = torch.cross(a, b)

    s = torch.norm(v, dim=-1)
    # mask zeros to prevent dicision by zero (entries with s=0 are zeroed eitherway because c=0 too)
    s[s == 0.] = -1.

    kmat = torch.zeros((bs, 3, 3), device=torch.device('cuda'))

    kmat[..., 0, 1] = -v[..., 2]
    kmat[..., 0, 2] = v[..., 1]
    kmat[..., 1, 0] = v[..., 2]
    kmat[..., 1, 2] = -v[..., 0]
    kmat[..., 2, 0] = -v[..., 1]
    kmat[..., 2, 1] = v[..., 0]

    # kmat = torch.tensor([[0., -v[..., 2], v[..., 1]],
    #                     [v[..., 2], 0., -v[..., 0]],
    #                     [-v[..., 1], v[..., 0], 0.]])

    R = torch.eye(3, device=torch.device('cuda')) + kmat + \
        (kmat@kmat) * ((1 - c) / (s ** 2))[..., None, None]

    return R


def align_neighborhood_batch(directions, neigb_directions):
    """
    align a neighborhood sampling such that the first direction
    is aligned with the respective vector in directions
    """

    assert directions.shape[-1] == 3
    assert neigb_directions.shape[0] > 1 and neigb_directions.shape[-1] == 3

    # TODO: catch [0, 0, 0] directions
    is_zerovec = torch.all(torch.eq(directions,
                           torch.tensor([[0., 0., 0.]]).cuda()), dim=-1)

    # put first vector as direction directly
    directions[is_zerovec] = neigb_directions[1]

    # normalize directions
    directions /= torch.clamp_min(torch.norm(directions, dim=-1, keepdim=True), epsilon)

    bs = directions.shape[0]
    neigb_dir_batch = torch.broadcast_to(neigb_directions, (bs,) + neigb_directions.shape)

    # align second neigb direction (first is center == [0, 0, 0])
    R = align_a_to_b(neigb_dir_batch[..., 1, :], directions)
    
    neigb_directions_r = torch.einsum('...ij,...nj->...ni', R, neigb_dir_batch)
    
    return neigb_directions_r



class EquivarianceChecker(torch.nn.Module):

    def __init__(self,
                 state_irreps='1x0e+1x2e+1x4e+1x0e',
                 n_fod=7,
                 n_dirs=8,
                 n_ico_subdiv=1,
                 use_graph=False,
                 graph_from_state=True):

        # get all xyz points
        super().__init__()
        xyz = get_ico_points(subdiv=n_ico_subdiv)
        alphas, betas = o3.xyz_to_angles(tt(xyz))

        self.rotm = o3.angles_to_matrix(alphas, betas, torch.zeros_like(alphas))

        if graph_from_state:
            self.state_irreps = state_irreps + '1x1o'
        else:
            self.state_irreps = state_irreps

        self.irr = o3.Irreps(self.state_irreps) * n_fod + o3.Irreps('1x1o') * n_dirs
        self.state_D = self.get_state_D(self.rotm)

        self.cos_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.use_graph = use_graph


    def get_state_D(self, rmats):
        state_D = torch.stack([
            self.irr.D_from_matrix(rmat.cpu()) for rmat in rmats
        ]).cuda()

        return state_D

    def forward(self, in_, model_):
        n_rot = self.state_D.shape[0]
        n_batch = in_.shape[0]

        # expand inputs and rotations
        in_exp = torch.repeat_interleave(in_, n_rot, dim=0)
        D_exp = self.state_D.repeat(n_batch, 1, 1)
        rotm_exp = self.rotm.repeat(n_batch, 1, 1)

        # get output from expanded unrotated inputs
        out_ = model_.to(torch.device('cuda'))(in_exp)
        # out_ = model_(in_exp)

        # rotate inputs
        in_r = torch.einsum('...i,...ji->...j', in_exp, D_exp)

        # rotate this output
        out_r = torch.einsum('...i,...ji->...j', out_, rotm_exp)

        # rotate graph if required
        if self.use_graph:
            assert hasattr(model_, 'rotate_graphs')
            assert hasattr(model_, '_get_graph')
            assert hasattr(model_, 'neighb_dirs')

            if model_.rotate_graphs:
                raise NotImplementedError

            graph = model_._get_graph(model_.neighb_dirs, n_batch * n_rot)
            
            # rotation only affects relative positions
            n_edges = (model_.neighb_dirs.shape[0] - 1) * 2
            graph.edata['rel_pos'] = torch.einsum('...i,...ji->...j',
                                            graph.edata['rel_pos'],
                                            torch.repeat_interleave(
                                                rotm_exp,
                                                n_edges,
                                                dim=0))

            out_rr = model_(in_r, graph)
        else:
            # get output from rotated inputs
            # out_rr = model_.to(torch.device('cuda'))(in_r)
            out_rr = model_(in_r)

        # compute equivariance measure
        cos_sim = self.cos_similarity(out_r, out_rr)

        return torch.mean(cos_sim)


def roto_transl(sl, rmat, shift, padshift=np.array([9., 0., 9.])):
    sl_rot = (sl - shift) @ rmat + shift

    # invert shift from padding
    sl_rot -= padshift

    return sl_rot

def roto_transl_inv(sl, rmat, shift, padshift=np.array([9., 0., 9.])):
    # invert shift from padding
    sl_pad = sl + padshift

    sl_rot = (sl_pad - shift) @ rmat.T + shift

    return sl_rot


def rotate_irr(irreps, rmat, x_r, return_np=False):
    if isinstance(rmat, np.ndarray):
        rmat_ = tt(rmat).cpu()
    else:
        rmat_ = rmat.cpu()

    if isinstance(x_r, np.ndarray):
        x_r_ = tt(x_r).cpu()
    else:
        x_r_ = x_r.cpu()

    irr = o3.Irreps(irreps)
    D_ = irr.D_from_matrix(rmat_.cpu())

    # rotate
    x_r_ = x_r_ @ tnp(D_.T)

    if return_np:
        return tnp(x_r_)
    return x_r_
