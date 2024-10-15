import pandas as pd


def clean_scores(df_):
    keep_labels = ['VC', 'IC', 'NC', 'mean_OL', 'mean_OR',
                   'mean_F1']
    drop_labels = [label for label in list(df_.columns.values)
                   if label not in keep_labels]
    df_.drop(columns=drop_labels, inplace=True)

def get_stats(df_):
    return df_.apply(lambda x: f'{x.mean()*100:.2f} +- {x.std()*100:.2f}',
                     axis=0)

def get_single(df_):
    return df_.apply(lambda x: f'{x.mean()*100:.2f}' , axis=0)



if __name__ == '__main__':
    # DEFINE PATHS FOR RESULTS
    td3_base = '/fabi_project/models/Exp3/exp1-td3-fix/1111/base_scores.csv'
    td3_rot = '/fabi_project/models/Exp3/exp1-td3-fix/1111/rotated_scores.csv'
    td3_basend = '/fabi_project/models/Exp3/exp1-td3-final_nodirinstate/1111/base_scores.csv'
    td3_rotnd = '/fabi_project/models/Exp3/exp1-td3-final_nodirinstate/1111/rotated_scores.csv'
    so3_base = '/fabi_project/models/Exp3/exp1-so3-fix/1111/base_scores.csv'
    so3_rot = '/fabi_project/models/Exp3/exp1-so3-fix/1111/rotated_scores.csv'

    outpath = '/fabi_project/experiments/Exp3/tables/rotated_scores'

    # LOAD SCORES
    td3_base_df = pd.read_csv(td3_base)
    td3_rot_df = pd.read_csv(td3_rot)
    td3_basend_df = pd.read_csv(td3_basend)
    td3_rotnd_df = pd.read_csv(td3_rotnd)
    so3_base_df = pd.read_csv(so3_base)
    so3_rot_df = pd.read_csv(so3_rot)

    # REMOVE UNWANTED COLUMNS
    clean_scores(td3_base_df)
    clean_scores(td3_rot_df)
    clean_scores(td3_basend_df)
    clean_scores(td3_rotnd_df)
    clean_scores(so3_base_df)
    clean_scores(so3_rot_df)

    # GET SERIES WITH THE STATS OR SINGLE VALUES FORMATTED
    td3_base_sr = get_single(td3_base_df)
    td3_rot_sr = get_stats(td3_rot_df)
    td3_basend_sr = get_single(td3_basend_df)
    td3_rotnd_sr = get_stats(td3_rotnd_df)
    so3_base_sr = get_single(so3_base_df)
    so3_rot_sr = get_stats(so3_rot_df)

    # COMBINE TO ONE DATAFRAME
    res = pd.DataFrame(
            [td3_base_sr, td3_rot_sr,
                  td3_basend_sr, td3_rotnd_sr,
                  so3_base_sr, so3_rot_sr],
            index=['td3 base scores', 'td3 rotated scores',
                   'td3 - nodir base scores', 'td3 - nodir rotated scores',
                   'so3 base scores', 'so3 rotated scores']
            )

    # WRITE TO DISK
    res.to_latex(outpath + '.tex')
    res.to_csv(outpath + '.csv')
