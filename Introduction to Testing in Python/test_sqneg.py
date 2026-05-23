import unittest
# Declaring the TestCase class
class TestSquared(unittest.TestCase):
    # Defining the test
    def test_negative(self):
        self.assertEqual((-3)**2, 9)
        