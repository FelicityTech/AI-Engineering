import unittest

def check_palindrome(string):
    reversed_string = string[::-1]
    return string == reversed_string

def create_data():
    return ['level', 'step', 'peep', 'toot']

class TestPalindrome(unittest.TestCase):
    def setUp(self):
        # Initialize data here
        self.data = create_data()
    
    def test_func(self):
        expected_result = [True, False, True, True]
        data_checked = list(map(check_palindrome, self.data))
        # Verify the checked data here
        self.assertEqual(data_checked, expected_result)

    def tearDown(self):
        # Clear the data here
        self.data.clear()
