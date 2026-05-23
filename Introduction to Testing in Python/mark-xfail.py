import pytest

def gen_sequence(n):
    return list(range(1, n+1))


# The xfail marker example

@pytest.mark.xfail
def test_gen_seq():
    assert gen_sequence(-1)