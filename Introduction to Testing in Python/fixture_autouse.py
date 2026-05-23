import pytest
import pandas as pd

# Autoused fixture

@pytest.fixture(autouse=True)
def set_pd_options():
    pd.set_option("display.max_columns", 5000)

# Test function
def test_pd_options():
    assert pd.get_option('display.max_columns') == 5000