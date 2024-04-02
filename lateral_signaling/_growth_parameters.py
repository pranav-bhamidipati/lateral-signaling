from dataclasses import dataclass
from os import PathLike
import pandas as pd


@dataclass
class MLEGrowthParams:
    """Stores growth parameters of logistic equation. Parameterized using MLE."""

    rho_max_ratio: float
    rho_max_inv_mm2: float
    g_inv_days: float

    @classmethod
    def from_csv(cls, params_csv: PathLike, reference_treatment: str = "10% FBS"):
        df = pd.read_csv(params_csv, index_col="treatment")
        rho_max = df.loc[reference_treatment, "rho_max_ratio"]
        rho_max_inv_mm2 = df.loc[reference_treatment, "rho_max_inv_mm2"]
        g_inv_days = df.loc[reference_treatment, "g_inv_days"]
        return cls(
            rho_max,
            rho_max_inv_mm2,
            g_inv_days,
        )

    @classmethod
    def empty(cls):
        kws = cls.__dataclass_fields__.keys()
        return cls(**{kw: None for kw in kws})

    def update_from_csv(
        self, params_csv: PathLike, reference_treatment: str = "untreated"
    ):
        df = pd.read_csv(params_csv, index_col="treatment")
        self.rho_max_ratio = df.loc[reference_treatment, "rho_max_ratio"]
        self.rho_max_inv_mm2 = df.loc[reference_treatment, "rho_max_inv_mm2"]
        self.g_inv_days = df.loc[reference_treatment, "g_inv_days"]
