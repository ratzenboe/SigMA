import copy
import numpy as np


def nanify_df(df, cols, percentage):
    """Randomly add NaNs to a dataframe in given columns"""
    df_copy = copy.deepcopy(df)
    size = df.shape[0]
    nans_size = int(size * percentage/100)
    # Randomly determine rows to drop
    idx2drop = np.random.choice(size, nans_size, replace=False)
    df_copy.loc[idx2drop, cols] = np.nan
    return df_copy
