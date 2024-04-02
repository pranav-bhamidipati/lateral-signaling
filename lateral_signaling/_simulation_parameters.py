from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass
class SimulationParameters:
    """Container for parameters used in signaling phase calculations."""

    alpha: float
    k: float
    p: float
    delta: float
    lambda_: float
    g: float
    rho_0: float
    beta_function: str
    beta_args: Iterable
    delay: float
    r_int: float
    gamma_R: float
    tmax_days: float
    nt_t: int
    nt_t_save: int
    rows: int
    cols: int
    dde_args: Iterable
    v_init_thresh: float

    @classmethod
    def from_json(cls, params_json: Path):
        with params_json.open("r") as f:
            j = json.load(f)
        return cls(**j)

    @classmethod
    def empty(cls):
        kws = cls.__dataclass_fields__.keys()
        return cls(**{kw: None for kw in kws})

    def update_from_json(self, params_json: Path):
        with params_json.open("r") as f:
            j = json.load(f)
        for k, v in j.items():
            setattr(self, k, v)
