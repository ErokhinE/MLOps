import os
import pandas as pd
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch
import sys
from src.data import read_datastore
import pandas as pd

def test_read_datastore():
    res = read_datastore()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], pd.DataFrame)