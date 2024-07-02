import pytest
import pandas as pd


@pytest.fixture(autouse=True)
def use_pandas_cow_behavior():
    pd.options.mode.copy_on_write = True
    yield
    pd.options.mode.copy_on_write = False
