from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path


data_dir = Path("../data/simulations")
phase_params_json = data_dir.joinpath("phase_threshold.json")


@dataclass(frozen=True)
class PhaseParameters:
    """Container for parameters used in signaling phase calculations."""

    rho_ON: float
    rho_OFF: float
    v_init_thresh: float = -1.0


def _initialize(params_json: Path):
    with params_json.open("r") as f:
        j = json.load(f, object_pairs_hook=OrderedDict)  # Loads entries in order

    return PhaseParameters(*j.values())
