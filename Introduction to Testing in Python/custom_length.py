import pytest

# Define the fixture for returning the length
@pytest.fixture
def list_length():
    return 10

# Define the fixture for a list preparation
@pytest.fixture
def prepare_list(list_length):
    return [i for i in range(list_length)]

def test_9(prepare_list):
    assert 9 in prepare_list
    assert 10 not in prepare_list
