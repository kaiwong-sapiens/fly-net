"""Paths shared by the pipeline scripts. Override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FLYNET_DATA", ROOT / "data"))          # cloned data repos (see get_data.sh)
WORK = Path(os.environ.get("FLYNET_WORK", ROOT / "work"))          # intermediate files
GAME_DATA = Path(os.environ.get("FLYNET_GAME_DATA", ROOT / "game" / "data"))
RESULTS = ROOT / "results"

GAIN_MAX = 150.0   # Hz: highest LC10a rate the game can request
W_MAX = 3.0        # highest multiplier training can apply to an LC10a output synapse


def load(dataset: str = "flywire"):
    """Return a flysim.Connectome for 'flywire' (FAFB v783) or 'malecns' (neuPrint, needs a token)."""
    import flysim
    if dataset == "flywire":
        return flysim.load_flywire(DATA, "783")
    if dataset == "malecns":
        import load_malecns
        return load_malecns.load(cache=WORK / "malecns")
    raise ValueError(f"unknown dataset {dataset!r}")
