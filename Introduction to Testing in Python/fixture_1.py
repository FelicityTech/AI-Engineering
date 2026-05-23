import pytest

# Fixture decorator
@pytest.fixture
# Fixture for data inintialization
def data():
    return [0,1,1,2,3,5,8,13,21]


def test_list(data):
    assert len(data) == 9
    assert 5 in data
    assert 21 in data