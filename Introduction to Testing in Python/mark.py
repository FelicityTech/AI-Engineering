import pytest

def get_length(string):
    return len(string)

# The test marker syntax
@pytest.mark.skip
def test_get_len():
    assert get_length('123') == 3