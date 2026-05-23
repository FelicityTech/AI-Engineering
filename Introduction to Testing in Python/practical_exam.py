import pandas as pd
import pytest, os

# Fixture to get the data
@pytest.fixture
def read_df():
    return pd.read_csv('ds_salaries.csv')
# Function to filter the data
def filter_df(df):
    return df[df['employment_type'] == 'FT']

# Function to get the mean
def get_mean(df):
    return df['salary_in_usd'].mean()


def test_read_df(read_df):
    # Check the typr of the dataframe
    assert isinstance(read_df, pd.DataFrame)
    # Check that df contains rows
    assert read_df.shape[0] > 0

# Integration test
def test_write():
    # Opening a file in writing mode
    with open('temp.txt', 'w') as wfile:
        # Writing the text to the file
        wfile.write('Testing stuff is awesome')
        # Check the file exists
        assert os.path.exists('temp.txt')
        # Don't forget to clean after yourself
        os.remove('temp.txt')

# Unit tests
def test_units(read_df):
    filtered = filter_df(read_df)
    assert filtered['employment_type'].unique() == ['FT']
    assert isinstance(get_mean(filtered), float)


# Feature test
def test_feature(read_df):
    # Filtering the data
    filtered = filter_df(read_df)
    # Test case: mean is greater than zero
    assert get_mean(filtered) > 0
    # Test case: mean is not biggere than the maximum
    assert get_mean(filtered) <= read_df['salary_in_usd'].max()

# Performance tests
def test_performance(benchmark, read_df):
    # Benchmark decorator
    @benchmark
    # Function to measure
    def get_result():
        filtered = filter_df(read_df)
        return get_mean(filtered)