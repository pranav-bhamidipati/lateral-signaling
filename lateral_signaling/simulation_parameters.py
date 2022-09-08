from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


_data_dir = Path("../data/simulations")
_simulation_params_json = _data_dir.joinpath("sim_parameters.json")


@dataclass(frozen=True)
class SimulationParameters:
    """Container for parameters used in signaling phase calculations."""

    alpha : float
    k : float
    p : float
    delta : float
    lambda_ : float
    g : float
    rho_0 : float
    beta_function : str
    beta_args : Iterable
    delay : float
    r_int : float
    gamma_R : float
    tmax_days : float
    nt_t : int
    nt_t_save : int
    rows : int
    cols : int
    dde_args : Iterable


def _initialize(params_json: Path):
    with params_json.open("r") as f:
        j = json.load(f)

    return SimulationParameters(**j)
