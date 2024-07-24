import os
import pandas as pd
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch
import sys

from hydra.experimental import initialize, compose
dir_path = os.environ['PROJECT_DIR']

@pytest.fixture(scope="session")
def config():
    with initialize(config_path=f"{dir_path}/configs", job_name="test_app"):
        cfg = compose(config_name="config", overrides=None)
        return cfg


def test_sample_data_creates_sample_file():
    # Check if the sampled file was created
    assert os.path.isfile(f'{dir_path}/data/samples/sample.csv')

def test_sample_data_content():
    df = pd.read_csv(f'{dir_path}/data/car_prices.csv')
    # Load the sampled file and check its contents
    sampled_df = pd.read_csv(f'{dir_path}/data/samples/sample.csv')
    assert not sampled_df.empty
    assert all(col in sampled_df.columns for col in df.columns)

