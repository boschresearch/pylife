import pytest
import pandas as pd


@pytest.fixture(autouse=True, scope="session")
def use_pandas_cow_behavior():
    if pd.__version__.startswith("2"):
        pd.options.mode.copy_on_write = True
    yield
    if pd.__version__.startswith("2"):
        pd.options.mode.copy_on_write = False
