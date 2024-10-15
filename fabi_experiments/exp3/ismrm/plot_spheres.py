import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata, Rbf
from TrackToLearn.utils.rotation import get_ico_points, tt, tnp
import e3nn.o3 as o3
from matplotlib.colors import Normalize


def load_data(fname, bname, col_name):
    base_df = pd.read_csv(bname)
    rot_df = pd.read_csv(fname)
    diff = np.abs(rot_df[col_name].values - base_df[col_name].values)

    return rot_df[col_name].values  # TODO test diff


if __name__ == '__main__':
    exp_root = '/fabi_project/experiments/Exp3/figures'

    xyz = get_ico_points(subdiv=1)
    alphas, betas = o3.xyz_to_angles(tt(xyz))
    angles = tnp(torch.stack([alphas, betas],)).T

    vmin, vmax = 0.0, .7

    # LOAD DATA
    base_scores = ['/fabi_project/models/Exp3/exp1-so3-fix_subd3/1111/base_scores.csv',
                   '/fabi_project/models/Exp3/exp1-td3-fix_subd3/1111/base_scores.csv']

    rot_scores = ['/fabi_project/models/Exp3/exp1-so3-fix_subd3/1111/rotated_scores'
                  '.csv',
                  '/fabi_project/models/Exp3/exp1-td3-fix_subd3/1111/rotated_scores'
                  '.csv']

    # TITLES ETC
    value_title = 'VC scores'
    subtitles = [f'{value_title} - so3 (proposed)', f'{value_title} - td3']

    # fig, axes = plt.subplots(nrows=2, ncols=1, subplot_kw={'projection': 'hammer'})
    fig, axes = plt.subplots(nrows=2, ncols=1)
    cmap = plt.cm.get_cmap('viridis')
    normalizer = Normalize(vmin, vmax)
    im = plt.cm.ScalarMappable(norm=normalizer)


    for bs, rs, ax, st in zip(base_scores, rot_scores, axes.flat, subtitles):

        val = load_data(rs, bs, 'VC')

        # shift angles (just radiants convention) (skipped)
        angles_ = angles.copy()
        # angles_[..., 1][angles[..., 1] > np.pi/2] = angles[..., 1][angles[...,
        # 1] > np.pi/2] - np.pi

        # init interpolation grid
        grid_x, grid_y = np.mgrid[-np.pi:np.pi:300j, 0.:np.pi:300j]

        # interpolate
        rbf3 = Rbf(angles[..., 0], angles_[..., 1], val, function="multiquadric",
                   smooth=0)
        z_lin = rbf3(grid_x.flatten(), grid_y.flatten())
        z_lin2 = griddata(angles_, val, (grid_x, grid_y),
                         method='nearest')

        # non-interpolated contour plot
        ax.contourf(grid_x, grid_y, z_lin2, 50, vmin=vmin, vmax=vmax)

        # plot also the coordinates where we have data
        ax.plot(angles_[..., 0], angles_[..., 1], 'or',  fillstyle='none')
        ax.plot(0., 0., 'or')  # no rotation

        # set axis to radians
        # Custom radian ticks and tick labels
        custom_xticks = [-np.pi, -np.pi/2, 0., np.pi/2, np.pi]
        custom_xtick_labels = ["$-\pi$", "$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"]
        ax.set_xticks(custom_xticks)
        ax.set_xticklabels(custom_xtick_labels)
        custom_yticks = [0., np.pi/2, np.pi]
        custom_ytick_labels = ["$0$", "$\pi/2$", "$\pi$"]
        ax.set_yticks(custom_yticks)
        ax.set_yticklabels(custom_ytick_labels)

        ax.title.set_text(st)
        ax.set_xlabel(r"$\alpha$ (radians)")
        ax.set_ylabel(r"$\beta$ (radians)")

    fig.set_size_inches(5, 5)

    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist()) # , orientation='horizontal')
    fig.savefig(os.path.join(exp_root, 'fig_exp1.png'), dpi=500,
                bbox_inches='tight')
    pass
