import pytest
import pandas as pd

@pytest.fixture
def data():
    df = pd.read_csv('/usr/local/share/games.csv')
    # Return df with the special keyword
    yield df
    # Remove all rows in df
    df.drop(df.index, inplace=True)
    # Delete the df variable
    del df

def test_type(data):
    assert type(data) == pd.DataFrame

def test_shape(data):
    assert data.shape[0] == 1512
