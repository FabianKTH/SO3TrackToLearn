import torch


TOL = 1e-6


def test_alignment():
    """
    check if batch of vector is aligned with second of neighbourhood
    directions
    """

    from TrackToLearn.environments.utils import get_neighborhood_directions
    from TrackToLearn.utils.rotation import align_neighborhood_batch

    batchsize = 32
    dirs = torch.randn((batchsize, 3), device=torch.device('cuda'))
    n_dirs = get_neighborhood_directions(radius=1., mode='star')
    n_dirs = torch.tensor(n_dirs, device=torch.device('cuda')).type(torch.float32)

    n_dirs_r = align_neighborhood_batch(dirs, n_dirs)

    # check if second n_dir is aligned
    dot = lambda x, y: torch.einsum('...i,...i->...', x, y)
    dres = dot(dirs, n_dirs_r[..., 1, :])

    assert torch.allclose(dres, torch.ones_like(dres), atol=TOL)


def test_alignment_with_zerodirs():
    """
    check if batch of vector is aligned with second of neighbourhood
    directions
    """

    from TrackToLearn.environments.utils import get_neighborhood_directions
    from TrackToLearn.utils.rotation import align_neighborhood_batch

    batchsize = 32
    dirs = torch.randn((batchsize, 3), device=torch.device('cuda'))

    # insert zeros
    rnd_idx = torch.randn(32) > 0.
    print(rnd_idx)
    dirs[rnd_idx] = torch.zeros(3, device=torch.device('cuda'))

    n_dirs = get_neighborhood_directions(radius=1., mode='star')
    n_dirs = torch.tensor(n_dirs, device=torch.device('cuda')).type(torch.float32)

    n_dirs_r = align_neighborhood_batch(dirs, n_dirs)

    # check if second n_dir is aligned
    dot = lambda x, y: torch.einsum('...i,...i->...', x, y)
    dres = dot(dirs, n_dirs_r[..., 1, :])

    assert torch.allclose(dres, torch.ones_like(dres), atol=TOL)


def check_all_ortho(vecs):
    n_vec = vecs.shape[0]
    dot = lambda x, y: torch.einsum('...i,...i->...', x, y)

    for i in range(n_vec):
        for j in range(n_vec):
            if i == j:
                continue

            print(f'{vecs[i]}, {vecs[j]}, {dot(vecs[i], vecs[j])}')

            assert (torch.allclose(dot(vecs[i], vecs[j]), torch.zeros(1).cuda(), atol=TOL) or
                    torch.allclose(dot(vecs[i], vecs[j]), -torch.ones(1).cuda(), atol=TOL))


def test_ortho():
    """
    check if a neighbourhood with orthogonal directions is still
    orthogonal after alignment
    """

    from TrackToLearn.environments.utils import get_neighborhood_directions
    from TrackToLearn.utils.rotation import align_neighborhood_batch

    batchsize = 32
    dirs = torch.randn((batchsize, 3), device=torch.device('cuda'))
    n_dirs = get_neighborhood_directions(radius=1., mode='star')
    n_dirs = torch.tensor(n_dirs, device=torch.device('cuda')).type(torch.float32)

    n_dirs_r = align_neighborhood_batch(dirs, n_dirs)

    # test orthogonal before
    check_all_ortho(n_dirs[1:])

    # test orthogonal after
    for b_ in range(batchsize):
        check_all_ortho(n_dirs_r[b_, 1:])


if __name__ == '__main__':
    test_alignment_with_zerodirs()
