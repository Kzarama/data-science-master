import pandas as pd


def get_cols_by_type(df):
    cat_cols = [df.dtypes.index[i_dtypes] for i_dtypes in range(
        0, len(df.dtypes)) if df.dtypes[i_dtypes] == "object"]
    num_cols = [df.dtypes.index[i_dtypes] for i_dtypes in range(
        0, len(df.dtypes)) if df.dtypes[i_dtypes] != "object"]
    return cat_cols, num_cols


def one_hot_encoder(df, categorical_cols, drop_first=False):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df
