import pytest
def multiple_of_two(num):
    if num == 0:
        raise(ValueError)
    return num % 2 == 0

def test_zero():    
  	# Add a context for an exception test here
    with pytest.raises(ValueError):
      	# Check zero input below
        multiple_of_two(num=0)