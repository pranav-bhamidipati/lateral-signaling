from dataclasses import dataclass
from os import PathLike
import pandas as pd


@dataclass(frozen=True)
class MLEGrowthParams:
    """Stores growth parameters of logistic equation. Parameterized using MLE."""

    rho_max_ratio: float
    rho_max_inv_mm2: float
    g_inv_days: float

    @classmethod
    def from_csv(cls, params_csv: PathLike):
        df = pd.read_csv(params_csv, index_col="treatment")
        rho_max = df.loc["untreated", "rho_max_ratio"]
        rho_max_inv_mm2 = df.loc["untreated", "rho_max_inv_mm2"]
        g_inv_days = df.loc["untreated", "g_inv_days"]
        return cls(
            rho_max,
            rho_max_inv_mm2,
            g_inv_days,
        )
