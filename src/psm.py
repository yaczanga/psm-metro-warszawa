"""Propensity score, dopasowanie najblizszym sasiadem i oszacowanie ATT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config


def standardise(features: pd.DataFrame) -> pd.DataFrame:
    """Standaryzuje zmienne kontrolne do sredniej zero i wariancji jeden.

    Algorytm najblizszego sasiada liczy odleglosc euklidesowa, wiec bez
    standaryzacji zmienne o duzych wartosciach (powierzchnia, rok budowy)
    zdominowalyby funkcje dystansu nad zmiennymi zero-jedynkowymi.
    """
    scaled = StandardScaler().fit_transform(features)
    return pd.DataFrame(scaled, columns=features.columns, index=features.index)


def propensity_score(scaled: pd.DataFrame, treatment: pd.Series,
                     model: str) -> np.ndarray:
    """Prawdopodobienstwo przynaleznosci do grupy eksperymentalnej.

    Oba warianty formy funkcyjnej sa deterministyczne: regresja logistyczna
    rozwiazywana jest algorytmem lbfgs bez losowego punktu startowego, probit
    metoda Newtona-Raphsona ze startem w zerze. Dwa przebiegi tego samego kodu
    daja ten sam wynik, wiec nie przekazujemy generatora losowego.
    """
    if model == "logit":
        estimator = LogisticRegression(
            max_iter=config.LOGIT_MAX_ITER, C=config.LOGIT_C
        )
        estimator.fit(scaled.to_numpy(), treatment)
        return estimator.predict_proba(scaled.to_numpy())[:, 1]
    if model == "probit":
        with_const = sm.add_constant(scaled)
        fitted = sm.Probit(treatment, with_const).fit(
            disp=0, maxiter=config.PROBIT_MAX_ITER
        )
        return fitted.predict(with_const).to_numpy()
    raise ValueError(f"Nieznana forma funkcyjna propensity score: {model!r}")


def match(scores: np.ndarray, model: pd.DataFrame, neighbours: int,
          caliper_sd=None) -> dict:
    """Dopasowuje do kazdego lokalu przy metrze najblizszych sasiadow.

    Dopasowanie odbywa sie ZE ZWRACANIEM: ten sam lokal kontrolny moze trafic
    do wielu par. Caliper odrzuca cala pare, gdy NAJDALSZY z przyjetych
    sasiadow wypada poza limitem, wyrazonym w odchyleniach standardowych
    samego propensity score.
    """
    frame = model.copy()
    frame["pscore"] = scores
    treated = frame[frame[config.TREATMENT] == 1]
    control = frame[frame[config.TREATMENT] == 0]

    finder = NearestNeighbors(n_neighbors=neighbours, metric="euclidean")
    finder.fit(control[["pscore"]])
    distances, indices = finder.kneighbors(treated[["pscore"]])

    if caliper_sd is None:
        accepted = np.ones(len(treated), dtype=bool)
        limit = None
    else:
        limit = float(caliper_sd * np.std(scores))
        accepted = distances[:, -1] <= limit

    return {
        "treated": treated[accepted].copy(),
        "control": control,
        "indices": indices[accepted],
        "neighbours": neighbours,
        "caliper_limit": limit,
        "dropped": int((~accepted).sum()),
    }


def synthetic_control(matched: dict, column: str) -> np.ndarray:
    """Wartosc sobowtora dla kazdej pary: srednia z dopasowanych sasiadow.

    Splaszczona lista indeksow wraca do macierzy par na sasiadow, wiec srednia
    liczona jest w kazdym wierszu osobno. Usrednienie calej listy dalo by jedna
    liczbe dla wszystkich par i zgubilo strukture parowa.
    """
    values = matched["control"].iloc[matched["indices"].flatten()][column].to_numpy()
    return values.reshape(-1, matched["neighbours"]).mean(axis=1)


def paired_differences(matched: dict, column: str) -> np.ndarray:
    """Roznica miedzy lokalem przy metrze a jego sobowtorem, para po parze."""
    return matched["treated"][column].to_numpy() - synthetic_control(matched, column)


def estimate(matched: dict) -> dict:
    """ATT w zlotych i procentach wraz z istotnoscia i przedzialem ufnosci.

    ATT jest srednia z roznic parowych, nie roznica srednich: przy dopasowaniu
    ze zwracaniem te dwie wielkosci nie sa sobie rowne. Istotnosc liczy test
    t-Studenta dla par, a przedzial ufnosci pochodzi z tego samego bledu
    standardowego roznic parowych.
    """
    differences_log = paired_differences(matched, config.LOG_OUTCOME)
    differences_pln = paired_differences(matched, config.OUTCOME)

    att_log = float(differences_log.mean())
    att_pln = float(differences_pln.mean())
    percent_effect = (np.exp(att_log) - 1) * 100

    _, p_value = stats.ttest_rel(
        matched["treated"][config.LOG_OUTCOME].to_numpy(),
        synthetic_control(matched, config.LOG_OUTCOME),
    )

    pairs = len(differences_log)
    standard_error = differences_log.std(ddof=1) / np.sqrt(pairs)
    half_width = config.CONFIDENCE_Z * standard_error
    lower = (np.exp(att_log - half_width) - 1) * 100
    upper = (np.exp(att_log + half_width) - 1) * 100

    return {
        "pary": pairs,
        "att_pln": att_pln,
        "att_log": att_log,
        "efekt_procentowy": float(percent_effect),
        "p_value": float(p_value),
        "blad_standardowy_log": float(standard_error),
        "ci_dolny_procent": float(lower),
        "ci_gorny_procent": float(upper),
    }


def run_scenario(scenario: dict, scaled: pd.DataFrame, treatment: pd.Series,
                 model: pd.DataFrame) -> dict:
    """Przeprowadza jeden wariant: estymacja, dopasowanie, oszacowanie."""
    scores = propensity_score(scaled, treatment, scenario["model"])
    matched = match(
        scores, model, scenario["neighbours"], scenario.get("caliper_sd")
    )
    result = estimate(matched)
    result["wariant"] = scenario["name"]
    result["scores"] = scores
    result["matched"] = matched
    return result


def run_all_scenarios(scaled: pd.DataFrame, treatment: pd.Series,
                      model: pd.DataFrame, scenarios=None) -> dict:
    """Przeprowadza wszystkie warianty i zwraca tabele porownawcza.

    Zwraca ramke zestawiajaca warianty oraz pelne wyniki kazdego z nich, bo
    rysunki diagnostyczne i analiza podgrup potrzebuja dopasowanych obserwacji,
    nie tylko liczb z tabeli.
    """
    scenarios = config.SCENARIOS if scenarios is None else scenarios
    from .data_load import raw_difference

    raw = raw_difference(model)
    results = {}
    rows = []
    for scenario in scenarios:
        result = run_scenario(scenario, scaled, treatment, model)
        results[scenario["name"]] = result
        rows.append(
            {
                "Wariant": scenario["name"],
                "N": len(model),
                "Różnica surowa (zł/m2)": round(raw),
                "ATT (zł/m2)": round(result["att_pln"]),
                "ATT (%)": round(result["efekt_procentowy"], 2),
                "95% CI dolny (%)": round(result["ci_dolny_procent"], 2),
                "95% CI górny (%)": round(result["ci_gorny_procent"], 2),
                "p": result["p_value"],
                "Liczba par": result["pary"],
            }
        )
    return {"table": pd.DataFrame(rows), "results": results, "raw_difference": raw}


def subgroup_effects(matched: dict, column=None) -> pd.DataFrame:
    """Efekt procentowy w podprobach wyznaczonych przez zmienna zero-jedynkowa.

    Podzial dotyczy lokali z grupy eksperymentalnej. Ich sobowtory pozostaja
    te same co w oszacowaniu glownym, wiec podproby rozkladaja efekt caly, a nie
    licza go od nowa na zawezonej probie.
    """
    column = config.MATERIAL_COLUMN if column is None else column
    differences = paired_differences(matched, config.LOG_OUTCOME)
    flags = matched["treated"][column].to_numpy()

    rows = []
    for value, label in ((1, "wielka płyta"), (0, "cegła i nowe budownictwo")):
        subset = differences[flags == value]
        rows.append(
            {
                "Podgrupa": label,
                "N": len(subset),
                "Efekt (%)": round(
                    float((np.exp(subset.mean()) - 1) * 100), 2
                ) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)
