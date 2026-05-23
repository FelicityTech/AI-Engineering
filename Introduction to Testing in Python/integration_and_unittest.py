import pytest
import pandas as pd

DF_PATH = "/usr/local/share/salaries.csv"
@pytest.fixture
def read_df():
    return pd.read_csv(DF_PATH)

def get_grouped(df):
    return df.groupby('work_year').agg({'salary': 'describe'})['salary']

def test_read_df(read_df):
    # Check the type of the dataframe
    assert isinstance(read_df, pd.DataFrame)
    # Check that read_df contains rows
    assert read_df.shape[0] > 0

def test_grouped(read_df):
    df = read_df
    salary_by_year = get_grouped(df)
    # Check the nulls here
    assert salary_by_year.isna().sum().sum() == 0
