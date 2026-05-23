import pytest

def factorial(n):
    if n == 0: return 1
    elif (type(n) == int):
        return n * factorial(n-1)
    else: return -1

# Test case: expected input
def test_regular():
    assert factorial(5) == 120

# Test case: zero input
def test_zero():
    assert factorial(0) == 1

# Test case: input of a wrong type
def test_str():
    assert factorial('5') == -1
