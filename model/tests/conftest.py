import pytest

from infoquality.hyperparameters import HyperParameters
from infoquality.model import Model

# A tiny randomly initialized model so tests run quickly on CPU
TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"


@pytest.fixture(scope="session")
def hp() -> HyperParameters:
    return HyperParameters(model=TINY_MODEL, max_len=16, num_classes=3, dropout=0.3)


@pytest.fixture(scope="session")
def label_map() -> dict:
    return {"low": 0, "medium": 1, "high": 2}


@pytest.fixture(scope="session")
def model(hp, label_map) -> Model:
    model = Model(hp, label_map=label_map)
    model.eval()
    return model
