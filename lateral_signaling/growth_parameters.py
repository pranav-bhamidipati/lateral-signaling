from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd


_growth_params_dir = Path("../data/growth_curves_MLE")
_growth_params_csv = _growth_params_dir.joinpath("growth_parameters_MLE.csv")


@dataclass(frozen=True)
class MLEGrowthParams:
    """Stores growth parameters of logistic equation. Parameterized using MLE."""

    rho_max_ratio: float
    rho_max_inv_mm2: float
    g_inv_days: float


def _initialize(params_csv: PathLike):
    """Read MLE growth parameters from file"""

    df = pd.read_csv(params_csv, index_col="treatment")
    rho_max = df.loc["untreated", "rho_max_ratio"]
    rho_max_inv_mm2 = df.loc["untreated", "rho_max_inv_mm2"]
    g_inv_days = df.loc["untreated", "g_inv_days"]
    return MLEGrowthParams(
        rho_max,
        rho_max_inv_mm2,
        g_inv_days,
    )
