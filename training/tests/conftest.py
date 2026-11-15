import sys
from pathlib import Path

# training scripts import their helpers as `common.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
