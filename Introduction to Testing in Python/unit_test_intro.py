# Function for sum of elements
def sum_of_arr(array:list) -> int:
    return sum(array)

# Test case 1: regular array
def test_regular():
    assert sum_of_arr([1, 2, 3]) == 6
    assert sum_of_arr([100, 150]) == 250

# Test Case 2: empty list
def test_empty():
    assert sum_of_arr([]) == 0

# Test Case 3: one number
def test_one_number():
    assert sum_of_arr([10]) == 10
    assert sum_of_arr([0]) == 0