import infoquality.predictor as predictor
import pytest


@pytest.fixture
def p():
    class Predictor:
        def __init__(self):
            pass


def test_batch(p):
    pr = predictor.Predictor
    assert len(list(pr.batch(p, ["a"] * 100, 20))) == 5
