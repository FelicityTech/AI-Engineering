import pytest
@pytest.fixture
def init_list():
    return []

@pytest.fixture(autouse=True)
def add_numbers_to_list(init_list):
    # Fixture Setup
    init_list.extend([i for i in range(10)])
    # Fixture output
    yield init_list
    # Teardown statement
    init_list.clear()

def test_9(init_list):
    assert 9 in init_list