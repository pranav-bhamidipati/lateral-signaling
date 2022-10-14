from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PhaseParameters:
    """Container for parameters used in signaling phase calculations."""

    rho_ON: float
    rho_OFF: float
    v_init_thresh: float = -1.0


def _initialize(params_json: Path, **kw):
    with params_json.open("r") as f:
        j = json.load(f)

    return PhaseParameters(**j, **kw)
