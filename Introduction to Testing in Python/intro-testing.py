import pytest

# A function to test
def division(a, b):
    return a/b

# A test function
def test_raise():
    with pytest.raises(ZeroDivisionError):
        division(a=26, b=0)

def squared(number):
    return number * number

def test_squared():
    assert squared(-2) == squared(2)
division(5, 0)
squared(6)