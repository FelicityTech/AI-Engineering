import pytest
def get_length(string):
    return len(string)

# The skipif marker example
@pytest.mark.skipif('2 * 2 == 5')
def test_get_len():
    assert get_length('abc') == 3