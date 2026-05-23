import pytest
import pandas as pd

DF_PATH = "/usr/local/share/salaries.csv"
@pytest.fixture
def read_df():
    return pd.read_csv(DF_PATH)

def get_grouped(df):
    return df.groupby('work_year').agg({'salary': 'describe'})['salary']

def test_feature_2022(read_df):
    salary_by_year = get_grouped(read_df)
    salary_2022 = salary_by_year.loc[2022, '50%']
    # Check the median type here
    assert isinstance(salary_2022, float)
    # Check the median is greater than zero
    assert salary_2022 > 0

# Use benchmark here
def test_reading_speed(benchmark):
    benchmark(pd.read_csv, DF_PATH)
