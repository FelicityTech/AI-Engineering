import pytest, os

@pytest.fixture
def setup_file():
    # Create temporary file
    file = 'test_file.txt'
    with open(file, 'w') as f1:
        f1.write("Test data 1")
    yield file
    os.remove(file)

def test_fs(setup_file):
    file = setup_file
    # Check that the file was created successfully
    assert os.path.exists(file)