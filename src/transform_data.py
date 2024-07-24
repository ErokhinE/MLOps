import giskard
from data_prepare import data_validation
import pandas as pd
from typing import Optional, Tuple
from data import transform
def transform_data(df: pd.DataFrame):
    """
    Transforms the input dataframe according to the specified configuration and options.
    @param df: Input DataFrame to transform

    @return tuple (X, y)
    """
    # Directly calling the transform function without passing a transformer_version parameter
    X, y = transform(df)
    return X, y