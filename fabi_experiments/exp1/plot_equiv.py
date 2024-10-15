import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick


if __name__ == '__main__':
    exp_root = '/fabi_project/experiments/Exp1'

    exp_list = [
        'exp1-td3-final_first50',
        'exp1-td3-final_nodirinstate_first50',
        'exp1-so3-final_first50',
    ]

    llabels = ['td3 - with peaks', 'td3 - no peaks', 'so3(proposed)']

    # plt.style.use('seaborn')

    ax = plt.gca()
    ax.set_prop_cycle('color', plt.cm.viridis(np.linspace(0.3, 0.9, 3)))
    # plt.yscale('log')
    # ax.set_xlim([0, 100])
    # ax.set_ylim([0, 1.1])

    for ie, (exp, llabel) in enumerate(zip(exp_list, llabels)):
        rlist = list()
        for seed in ['1111', '2222', '3333', '4444', '5555']:
            res = pd.read_csv(os.path.join(exp_root,
                                           exp,
                                           f'{seed}/all_scores.csv'))

            # ep_ = [e_ for e_ in range(5, 55, 5)]
            # x_ = [1 - res.loc[res['actor_name'] == f'training_ep-{e_}',
            # 'equiv'].item()
            #       for e_ in ep_]
            res = res[res.actor_name.str.startswith('training_ep')]
            res['training_ep'] = res['actor_name'].apply(lambda x: int(x.split(
                    '-')[-1]))

            res['equiv_err'] = res['equiv'].apply(lambda x: 1 - x)

            rlist.append(res)

        sns.lineplot(pd.concat(rlist),
                     x='training_ep',
                     y='equiv_err',
                     ax=ax,
                     # style='whitegrid',
                     dashes=False,
                     markers=True,
                     markeredgecolor='w',
                     label=llabel,
                     )
    sns.set_style("whitegrid")
    plt.xlabel(None)
    handles, labels = ax.get_legend_handles_labels()
    for i in handles:
        i.set_markeredgecolor('w')
    legend = ax.legend(handles=handles[1:], labels=labels[1:])
    ax.legend()
    # plt.show()

    plt.xlabel('Training step')
    plt.ylabel(r'Equivariance error ($\uparrow$)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # sns.color_palette("viridis")

    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    fig.savefig(os.path.join(exp_root, 'fig_exp1.png'), dpi=500, bbox_inches='tight')

