"""Wspolliniowosc zmiennych kontrolnych: VIF i macierz korelacji."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config


def vif_table(features: pd.DataFrame) -> pd.DataFrame:
    """Czynnik inflacji wariancji dla kazdej zmiennej kontrolnej, malejaco.

    Liczony na probie modelowej, czyli po odrzuceniu strefy buforowej - to ten
    zbior wchodzi do estymacji propensity score. Stala jest dodawana przed
    obliczeniem i usuwana z wyniku: bez wyrazu wolnego VIF myli
    wspolliniowosc z odchyleniem od zera.
    """
    with_const = sm.add_constant(features.astype(float))
    table = pd.DataFrame(
        {
            "Zmienna": with_const.columns,
            "VIF": [
                variance_inflation_factor(with_const.to_numpy(), i)
                for i in range(with_const.shape[1])
            ],
        }
    )
    table = table[table["Zmienna"] != "const"]
    return table.sort_values("VIF", ascending=False).reset_index(drop=True)


def vif_summary(table: pd.DataFrame) -> dict:
    """Najwyzszy VIF, zmienna go osiagajaca i rozstrzygniecie wobec progu."""
    top = table.iloc[0]
    return {
        "max_vif": float(top["VIF"]),
        "zmienna": str(top["Zmienna"]),
        "prog": config.VIF_LIMIT,
        "ponad_progiem": int((table["VIF"] > config.VIF_LIMIT).sum()),
    }


def correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Macierz korelacji Pearsona miedzy zmiennymi kontrolnymi."""
    return features.corr()


def strongest_correlations(matrix: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    """Pary zmiennych o najwyzszej korelacji bezwzglednej.

    Przekatna i gorny trojkat sa pomijane, bo nie niosa nowej informacji.
    """
    pairs = matrix.where(
        ~pd.DataFrame(
            [[j >= i for j in range(len(matrix))] for i in range(len(matrix))],
            index=matrix.index,
            columns=matrix.columns,
        )
    ).stack()
    pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
    frame = pairs.head(count).reset_index()
    frame.columns = ["Zmienna A", "Zmienna B", "Korelacja"]
    return frame
