import unittest

class TestWord(unittest.TestCase):
    # Fixture setup method
    def setUp(self):
        # Initialize the word banana here
        self.word = 'banana'

    # Test method
    def test_the_word(self):
        # Add the tests here
        self.assertNotIn('B', self.word)
        self.assertNotIn('y', self.word)
        self.assertIn('b', self.word)

    
    # Fixture teardown method
    def tearDown(self):
        # Delete the word variable here
        del self.word
