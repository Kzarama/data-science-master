"""Custom functions for data preprocessing"""
import pandas as pd


def get_cols_by_type(df):
    """Get columns by type

    Args:
        df (pandas.DataFrame): DataFrame

    Returns:
        tuple: Columns by type
    """
    cat_cols = [df.dtypes.index[i_dtypes] for i_dtypes in range(
        0, len(df.dtypes)) if df.dtypes[i_dtypes] == "object"]
    num_cols = [df.dtypes.index[i_dtypes] for i_dtypes in range(
        0, len(df.dtypes)) if df.dtypes[i_dtypes] != "object"]
    return cat_cols, num_cols


def one_hot_encoder(df, categorical_cols, drop_first=False):
    """One hot encoder

    Args:
        df (pandas.DataFrame): DataFrame
        categorical_cols (list): List of categorical columns
        drop_first (bool): Drop first category

    Returns:
        pandas.DataFrame: One hot encoded DataFrame
    """
    df_encoded = pd.get_dummies(
        df, columns=categorical_cols, drop_first=drop_first)
    return df_encoded
