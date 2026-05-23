import pandas as pd
import pytest

df = pd.read_csv('ds_salaries.csv')

# Filter feature
def filter_data_by_exper(df, experience_name):
    filtered_df = df[df['experience_level'] == experience_name]
    return filtered_df


# Feature test function
def test_unique():
    exper_name = 'SE'   # Senior-level
    filtered = filter_data_by_exper(df, exper_name)

    assert filtered['experience_level'].nunique() == 1
    assert filtered['experience_level'].unique()[0] == exper_name