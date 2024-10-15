import numpy as np
import torch

TOL = 1e-3


def rot_z(gamma):
    return torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def _get_actor(input_irrep='1x0e + 1x2e + 1x4e'):
    from TrackToLearn.algorithms.so3_actor import SO3Actor

    # initialize model for testing the method
    return SO3Actor(spheredir='/fabi_project/sphere', input_irrep=input_irrep, hidden_lmax=4, hidden_depth=4,
                    add_neighborhood_vox=0.33, neighborhood_mode='star', n_dirs=8)


def _get_fod_state(batchsize=32, return_lastdir=False):

    # define components
    n_fod_coeff = 15  # lmax = 4, even order only
    n_fod = 7
    n_dirs = 8

    fods = torch.randn(batchsize, n_fod, n_fod_coeff)
    masks = torch.randn(batchsize, n_fod, 1)
    dirs = torch.randn(batchsize, n_dirs, 3)

    # format state from components
    signal = torch.cat([fods, masks], dim=-1).reshape(batchsize, -1)
    dirs_flat = dirs.reshape(batchsize, -1)

    state = torch.cat([signal, dirs_flat], dim=-1)

    # format fiberdict from components
    fd = {}
    fd['0'] = torch.cat([fods[..., 0, None], masks], dim=-1).reshape(-1, 2, 1)
    fd['1'] = torch.repeat_interleave(dirs, n_fod, dim=0)
    fd['2'] = fods[..., 1:6].reshape(-1, 1, 5)
    fd['4'] = fods[..., 6:].reshape(-1, 1, 9)

    # put the whole thing onto the gpu
    for k in fd.keys():
        fd[k] = fd[k].cuda()

    if return_lastdir:
        return state, fd, dirs[:, -1].cuda()
    else:
        return state, fd


def _get_peaks_state(batchsize=32, return_lastdir=False):

    # define components
    n_peaks = 10
    n_fod = 7
    n_dirs = 8

    peaks = torch.randn(batchsize, n_fod, n_peaks * 3)
    masks = torch.randn(batchsize, n_fod, 1)
    dirs = torch.randn(batchsize, n_dirs, 3)

    # format state from components
    signal = torch.cat([peaks, masks], dim=-1).reshape(batchsize, -1)
    dirs_flat = dirs.reshape(batchsize, -1)

    state = torch.cat([signal, dirs_flat], dim=-1)

    # format fiberdict from components
    fd = {}
    fd['0'] = masks.reshape(-1, 1, 1)
    # fd['1'] = torch.repeat_interleave(dirs, n_fod, dim=0) + fods.reshape(-1, 1, no_fod_coeff)
    fd['1'] = torch.cat([peaks.reshape(-1, n_peaks, 3),
                         torch.repeat_interleave(dirs, n_fod, dim=0)],
                        dim=1)

    # put the whole thing onto the gpu
    for k in fd.keys():
        fd[k] = fd[k].cuda()

    if return_lastdir:
        return state, fd, dirs[:, -1].cuda()
    else:
        return state, fd


def test_state_format_fod():
    act = _get_actor()
    state, fd = _get_fod_state()

    # call state formatter
    fd_out, _ = act._reformat_state(state.cuda())

    # test if all entries are the same
    _check_fiberdict(fd, fd_out)


def _check_fiberdict(fd, fd_out):
    for k in fd.keys():
        print(f'shape {fd[k].shape} and {fd_out[k].shape}')
        assert torch.allclose(fd[k].cuda(), fd_out[k], atol=TOL), (
                f'degree {k}: {fd[k]} and {fd_out[k]}')


def _rotate_state(state, rot, state_irreps='1x0e+1x2e+1x4e+1x0e', n_fod=7, n_dirs=8):
    import e3nn
    irr = e3nn.o3.Irreps(state_irreps)*n_fod + e3nn.o3.Irreps('1x1o')*n_dirs

    D = irr.D_from_matrix(rot.cpu())

    return state @ D.T.cuda()


def _rotate_fiberdict(fd, rot):
    import e3nn

    def get_irr(mul, l):
        parity = 'e' if l % 2 == 0 else 'o'
        return e3nn.o3.Irreps(f'{l}{parity}') * mul

    fd_ret = {}
    for k in fd.keys():
        sh_ = fd[k].shape
        mul_, l_ = sh_[1], int(k)

        D = get_irr(mul_, l_).D_from_matrix(rot.cpu())
        fd_k_flat = fd[k].reshape(sh_[0], -1)
        fd_ret[k] = (fd_k_flat.cuda() @ D.T.cuda()).reshape(*sh_)

    return fd_ret


def test_model_fod_equiv():
    batchsize = 32

    act = _get_actor()
    state, fd = _get_fod_state()
    neigh = act.neighb_dirs
    graph = act._get_graph(neigh, batchsize=batchsize)

    R = rot(*torch.rand(3)).cuda()
    state_r = _rotate_state(state.cuda(), R).cuda()
    fd_r = _rotate_fiberdict(fd, R)

    fd_rr, _ = act._reformat_state(state_r)
    _check_fiberdict(fd_r, fd_rr)

    graph_r = act._get_graph(neigh @ R.T, batchsize=batchsize)

    # model call
    out = act.cuda().model(graph, fd, {})
    out_r = act.cuda().model(graph_r, fd_r, {})

    # check
    assert torch.allclose(out @ R.T, out_r, atol=TOL)


def test_model_peak_equiv():
    batchsize = 32

    act = _get_actor(input_irrep='10x1o')
    state, fd = _get_peaks_state()
    neigh = act.neighb_dirs
    graph = act._get_graph(neigh, batchsize=batchsize)

    R = rot(*torch.rand(3)).cuda()
    state_r = _rotate_state(state.cuda(), R, state_irreps='10x1o + 1x0e').cuda()
    fd_r = _rotate_fiberdict(fd, R)

    fd_rr, _ = act._reformat_state(state_r)
    _check_fiberdict(fd_r, fd_rr)

    graph_r = act._get_graph(neigh @ R.T, batchsize=batchsize)

    # model call
    out = act.cuda().model(graph, fd, {})
    out_r = act.cuda().model(graph_r, fd_r, {})

    # check
    assert torch.allclose(out @ R.T, out_r, atol=TOL)


def test_model_equiv_rotate_neigh():
    batchsize = 32

    act = _get_actor()
    state, fd, lastdir = _get_fod_state(return_lastdir=True)
    neigh = act.neighb_dirs
    graph = act._get_graph(neigh, batchsize=batchsize, dirs=lastdir)

    R = rot(*torch.rand(3)).cuda()
    state_r = _rotate_state(state.cuda(), R).cuda()
    fd_r = _rotate_fiberdict(fd, R)

    fd_rr, _ = act._reformat_state(state_r)
    _check_fiberdict(fd_r, fd_rr)

    # Note: should also work with just neigh istead of neigh @ R.T. but due to axial degree of freedom doesnt.
    graph_r = act._get_graph(neigh @ R.T, batchsize=batchsize, dirs=lastdir @ R.T)

    # model call
    out = act.cuda().model(graph, fd, {})
    out_r = act.cuda().model(graph_r, fd_r, {})

    # check
    assert torch.allclose(out @ R.T, out_r, atol=TOL)


def test_model_equiv_spherepoints():
    import TrackToLearn.utils.rotation as rutil
    
    # setup test variables
    batchsize = 32
    act = _get_actor()
    state, fd = _get_fod_state()
    neigh = act.neighb_dirs
    graph = act._get_graph(neigh, batchsize=batchsize)

    # setup equivariance tester
    echeck = rutil.EquivarianceChecker()

    # checking
    equiv_measure = echeck(state.cuda(), act, graph)

    assert torch.allclose(equiv_measure, torch.tensor([1.]).cuda())


if __name__ == '__main__':
    test_model_equiv_spherepoints()
    # test_model_fod_equiv()
