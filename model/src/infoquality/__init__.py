from importlib.metadata import PackageNotFoundError, version

from infoquality.hyperparameters import HyperParameters
from infoquality.model import Model
from infoquality.predictor import Prediction, Predictor

try:
    __version__ = version("infoquality")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["HyperParameters", "Model", "Prediction", "Predictor", "__version__"]
