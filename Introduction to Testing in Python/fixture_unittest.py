import unittest

class TestLi(unittest.TestCase):
    # Fixture setup method
    def setUp(self):
        self.li = [i for i in range(100)]

    # Fixture teardown method
    def tearDown(self):
        self.li.clear()

    # Test method
    def test_your_list(self):
        self.assertIn(99, self.li)
        self.assertNotIn(100, self.li)
