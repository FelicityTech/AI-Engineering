import unittest
import pandas as pd

DF_PATH = 'https://assets.datacamp.com/production/repositories/6253/datasets/f015ac99df614ada3ef5e011c168054ca369d23b/energy_truncated.csv'

def get_data():
    return pd.read_csv(DF_PATH)

def min_country(df):
    return df['VALUE'].idxmin()

class TestDF(unittest.TestCase):

    def setUp(self):
        self.df = get_data()
        self.df.drop('previousYearToDate', axis=1, inplace=True)
        self.df = self.df.groupby('COUNTRY').agg({'VALUE': 'sum'})

    def test_NAs(self):
        # Check dataset contains no nulls
        self.assertEqual(self.df.isna().sum().sum(), 0)

    def test_min_country(self):
        # Check that min_country returns a string
        result = min_country(self.df)
        self.assertIsInstance(min_country(self.df), str)

    def tearDown(self):
        self.df.drop(self.df.index, inplace=True)