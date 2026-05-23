import pytest

@pytest.fixture
def setup_data():
    return [1,2,3,4,5]

@pytest.fixture
def process_data(setup_data):
    return [el * 2 for el in setup_data]

def test_process_data(process_data):
    assert process_data == [2, 4, 6, 8, 10]