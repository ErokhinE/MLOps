from src.data import validate_initial_data
def test_validation():
    assert isinstance(validate_initial_data(), bool)