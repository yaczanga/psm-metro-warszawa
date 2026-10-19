"""Standaryzowana roznica srednich (SMD) i wykres Love'a.

SMD mierzy, o ile odchylen standardowych rozjezdzaja sie srednie zmiennej
kontrolnej miedzy grupami. Wzor przyjety w badaniu:

    SMD = |X_T,m - X_C,m| / sqrt((S2_T,o + S2_C,o) / 2)

Licznik liczony jest na grupach DOPASOWANYCH, mianownik na grupach
PIERWOTNYCH, niedopasowanych. Stale odniesienie w mianowniku jest tu
istotne: dopasowanie samo z siebie zmniejsza wariancje w probie, wiec
mianownik liczony po dopasowaniu zawyzalby SMD i udawal pogorszenie balansu
tam, gdzie zaszla poprawa (Stuart, 2010).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .psm import synthetic_control


def _pooled_denominator(treated_raw: pd.Series, control_raw: pd.Series) -> float:
    """Pierwiastek ze sredniej wariancji w grupach pierwotnych."""
    variance_treated = treated_raw.var(ddof=1)
    variance_control = control_raw.var(ddof=1)
    return float(np.sqrt((variance_treated + variance_control) / 2))


def smd_table(model: pd.DataFrame, matched: dict, columns=None) -> pd.DataFrame:
    """SMD kazdej zmiennej kontrolnej przed i po dopasowaniu.

    Licznik po dopasowaniu to srednia grupy eksperymentalnej po nalozeniu
    calipera wobec sredniej jej sobowtorow, a mianownik pochodzi z grup
    pierwotnych. Zmienna o zerowej wariancji w obu grupach dostaje SMD rowny
    zero.
    """
    columns = (
        list(matched["treated"].columns.intersection(_control_columns(model)))
        if columns is None
        else list(columns)
    )

    treated_raw = model[model[config.TREATMENT] == 1]
    control_raw = model[model[config.TREATMENT] == 0]

    rows = []
    for column in columns:
        denominator = _pooled_denominator(treated_raw[column], control_raw[column])
        if denominator < 1e-12:
            smd_before = smd_after = 0.0
        else:
            smd_before = abs(
                treated_raw[column].mean() - control_raw[column].mean()
            ) / denominator
            smd_after = abs(
                matched["treated"][column].to_numpy().mean()
                - synthetic_control(matched, column).mean()
            ) / denominator
        rows.append(
            {
                "Zmienna": column,
                "SMD przed": float(smd_before),
                "SMD po": float(smd_after),
            }
        )

    table = pd.DataFrame(rows)
    table["Redukcja (%)"] = np.where(
        table["SMD przed"] > 1e-12,
        100 * (1 - table["SMD po"] / table["SMD przed"]),
        np.nan,
    )
    return table.sort_values("SMD przed", ascending=False).reset_index(drop=True)


def _control_columns(model: pd.DataFrame) -> list[str]:
    """Nazwy zmiennych kontrolnych w ramce proby modelowej."""
    return [
        c
        for c in model.columns
        if c not in config.EXCLUDED_FROM_FEATURES and c != "pscore"
    ]


def balance_summary(table: pd.DataFrame) -> dict:
    """Liczby opisujace jakosc zbalansowania, gotowe do tekstu i do testow."""
    return {
        "zmiennych": len(table),
        "max_smd_przed": float(table["SMD przed"].max()),
        "max_smd_po": float(table["SMD po"].max()),
        "zmienna_max_po": table.loc[table["SMD po"].idxmax(), "Zmienna"],
        "ponad_limitem": int((table["SMD po"] > config.SMD_LIMIT).sum()),
        "ponad_progiem": int((table["SMD po"] > config.SMD_TARGET).sum()),
        "ponizej_progu": int((table["SMD po"] <= config.SMD_TARGET).sum()),
    }
