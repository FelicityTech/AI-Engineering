def multiple_of_two(num):
    if num == 0:
        raise(ValueError)
    return num % 2 == 0

# Add the pytest marker decorator here
@pytest.mark.xfail
def test_fails():
    # Write any assert test that will fail
    assert multiple_of_two(2) == 0