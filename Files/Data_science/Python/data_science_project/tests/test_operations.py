from src.models.operations import Operations


def test_calculate_average():
    assert Operations.calculate_average([2, 2, 6, 6]) == 4