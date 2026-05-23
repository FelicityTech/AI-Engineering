class Rectangle:
    # Constructor of Rectangle
    def __init__(self, a, b):
        self.a = a
        self.b = b

    # Area method
    def get_area(self):
        return self.a * self.b

# Usage example
r = Rectangle(4, 5)
print(r.get_area())


# Inheritance
class RedRectangle(Rectangle):
    self.color = 'red'


import unittest
# Declaring the TestCase class
class TestSquared(unittest.TestCase):
    # Defining the test
    def test_negative(self):
        self.assertEqual((-3) ** 2, 9)