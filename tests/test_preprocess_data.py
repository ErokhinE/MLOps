import os
import pandas as pd
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch
import sys
from src.data import preprocess_data


def test_preprocess_data():
    dir_path = os.environ['PROJECT_DIR']
    df = pd.read_csv(f'{dir_path}/data/samples/sample.csv')
    res = preprocess_data(df)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], pd.DataFrame)
    assert isinstance(res[1], pd.DataFrame)
    assert len(res[1]) == len(res[0])