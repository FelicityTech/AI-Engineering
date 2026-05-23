import pytest
# Fixture that is requested by the other fixture
@pytest.fixture
def setup_data():
    return "I am a fixture!"

# Fixture that is requested by the test function
@pytest.fixture
def process_data(setup_data):
    return setup_data.upper()

# The test function
def test_process_data(process_data):
    assert process_data == "I AM A FIXTURE!"

