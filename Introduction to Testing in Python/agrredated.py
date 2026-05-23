# Don't forget to run
# pytest agg_with_sum.py 
# in the CLI to actually run the test!

import pandas as pd
import pytest

# Fixture to prepare the data
@pytest.fixture
def get_df():
    return pd.read_csv('https://assets.datacamp.com/production/repositories/6253/datasets/757c6cb769f7effc5f5496050ea4d73e4586c2dd/laptops_train.csv')

# Aggregation feature
def agg_with_sum(data, group_by_column, aggregate_column):
    return data.groupby(group_by_column)[aggregate_column].sum()

# Test function
def test_agg_feature(get_df):
    # Aggregate preparation
    aggregated = agg_with_sum(get_df, 'Manufacturer', 'Price')
    # Test the type of the aggregated
    assert type(aggregated) == pd.Series
    # Test the number of rows of the aggregated
    assert len(aggregated) > 0
    # Test the data type of the aggregated
    assert aggregated.dtype in (int, float)
