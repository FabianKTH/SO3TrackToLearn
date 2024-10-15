import torch

device = torch.device('cuda')


def tt(x, dtype=torch.float32):
    """
    tt: to torch, shorthand for converting numpy to torch
    """

    return torch.from_numpy(x).to(device=device, dtype=dtype)


def tnp(x: torch.Tensor):
    return x.detach().cpu().numpy()
