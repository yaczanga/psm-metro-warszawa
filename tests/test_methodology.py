"""Gwarancje metodologiczne badania.

Testy nie sprawdzaja, czy kod sie wykonuje, tylko czy nie lamie zalozen, na
ktorych opiera sie wniosek: wykluczenia strefy buforowej, zbalansowania proby
po dopasowaniu, integralnosci par i zgodnosci efektu procentowego z ATT
liczonym na logarytmach.
"""

import numpy as np
import pandas as pd
import pytest

from src import balance, config, data_load, psm


# --------------------------------------------------------------------------
# Proba rzeczywista - liczona raz na caly modul, bo przebieg trwa kilka sekund
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample():
    return data_load.build_model_sample()


@pytest.fixture(scope="module")
def main_scenario(sample):
    scaled = psm.standardise(sample["features"])
    scenario = next(
        s for s in config.SCENARIOS if s["name"] == config.MAIN_SCENARIO
    )
    return psm.run_scenario(scenario, scaled, sample["treatment"], sample["model"])


# --------------------------------------------------------------------------
# Pomoce do prob syntetycznych
# --------------------------------------------------------------------------
def _synthetic_matched(treated_values, control_values, indices, neighbours,
                       extra_treated=None, extra_control=None):
    """Sklada strukture dopasowania z podanych cen, bez uruchamiania estymacji.

    Pozwala sprawdzic sam rachunek ATT na liczbach, ktorych wynik znamy
    z recznego przeliczenia.
    """
    treated = pd.DataFrame({
        config.OUTCOME: np.exp(treated_values),
        config.LOG_OUTCOME: treated_values,
        config.TREATMENT: 1,
    })
    if extra_treated is not None:
        for name, values in extra_treated.items():
            treated[name] = values
    control = pd.DataFrame({
        config.OUTCOME: np.exp(control_values),
        config.LOG_OUTCOME: control_values,
        config.TREATMENT: 0,
    })
    if extra_control is not None:
        for name, values in extra_control.items():
            control[name] = values
    return {
        "treated": treated,
        "control": control,
        "indices": np.asarray(indices),
        "neighbours": neighbours,
        "caliper_limit": None,
        "dropped": 0,
    }


# --------------------------------------------------------------------------
# 1. Strefa buforowa
# --------------------------------------------------------------------------
def test_strefa_buforowa_wyklucza_przedzial_posredni():
    """Zaden lokal z przedzialu 0,5-1,2 km nie wchodzi do proby modelowej.

    Przedzial wypada w calosci, a zmienna interwencyjna rozdziela to, co
    zostalo, dokladnie na progu bliskosci. Test zawiedzie, gdyby drugi warunek
    filtra odwolywal sie do progu bliskosci zamiast do progu oddalenia -
    wtedy strefa buforowa wrocilaby do proby jako grupa kontrolna.
    """
    frame = pd.DataFrame({
        config.DISTANCE: [0.1, 0.49, 0.5, 0.8, 1.2, 1.21, 3.0],
        config.OUTCOME: np.arange(7, dtype=float) + 1,
    })
    result = data_load.apply_buffer_zone(frame)

    in_buffer = result[config.DISTANCE].between(
        config.CLOSE_KM, config.FAR_KM, inclusive="both"
    )
    assert not in_buffer.any(), "Strefa buforowa nie zostala wykluczona."
    assert list(result[config.DISTANCE]) == [0.1, 0.49, 1.21, 3.0]
    assert list(result[config.TREATMENT]) == [1, 1, 0, 0]


def test_proba_modelowa_nie_zawiera_strefy_buforowej(sample):
    """Gwarancja sprawdzona takze na danych rzeczywistych, nie tylko na atrapie.

    Licznosci nie sa wpisane w test - pochodza z tej samej ramki, ktora liczy
    pipeline, wiec test pilnuje struktury proby, a nie zapamietanych liczb.
    """
    model = sample["model"]
    distances = model[config.DISTANCE]
    assert not distances.between(config.CLOSE_KM, config.FAR_KM).any()

    treated = model[config.TREATMENT] == 1
    assert (distances[treated] < config.CLOSE_KM).all()
    assert (distances[~treated] > config.FAR_KM).all()
    assert int(treated.sum()) == sample["counts"]["grupa_eksperymentalna"]
    assert len(model) == sample["counts"]["proba_modelowa"]


# --------------------------------------------------------------------------
# 2. Zbalansowanie po dopasowaniu
# --------------------------------------------------------------------------
def test_smd_po_dopasowaniu_ponizej_granicy(sample, main_scenario):
    """Zadna zmienna kontrolna nie przekracza granicy zbalansowania.

    To jest warunek, pod ktorym porownanie grup w ogole cokolwiek szacuje:
    gdyby ktorakolwiek zmienna zostala niezbalansowana, roznica cen nadal
    zawieralaby efekt tej zmiennej.
    """
    table = balance.smd_table(sample["model"], main_scenario["matched"])

    assert len(table) == sample["counts"]["zmienne_kontrolne"]
    przekroczone = table[table["SMD po"] > config.SMD_LIMIT]
    assert przekroczone.empty, (
        f"Zmienne powyzej granicy {config.SMD_LIMIT}: "
        f"{list(przekroczone['Zmienna'])}"
    )


def test_mianownik_smd_pochodzi_z_grup_pierwotnych():
    """Mianownik SMD liczony jest przed dopasowaniem, nie po nim.

    Dopasowanie zmniejsza wariancje w probie, wiec mianownik liczony po nim
    bylby mniejszy, a SMD wyzszy. Test buduje przypadek, w ktorym obie grupy
    dopasowane maja wariancje zerowa: gdyby mianownik pochodzil stamtad,
    rachunek nie mialby sie o co oprzec. Oczekiwana wartosc policzona jest
    recznie z wariancji grup pierwotnych.
    """
    model = pd.DataFrame({
        config.TREATMENT: [1, 1, 1, 1, 0, 0, 0, 0],
        "x": [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0],
        config.OUTCOME: np.ones(8),
        config.LOG_OUTCOME: np.zeros(8),
    })
    matched = _synthetic_matched(
        treated_values=np.array([0.0, 0.0]),
        control_values=np.array([0.0, 0.0]),
        indices=[[0], [1]],
        neighbours=1,
        extra_treated={"x": [3.0, 3.0]},
        extra_control={"x": [1.0, 1.0]},
    )

    table = balance.smd_table(model, matched, columns=["x"])
    wariancja = pd.Series([0.0, 2.0, 4.0, 6.0]).var(ddof=1)
    oczekiwany = abs(3.0 - 1.0) / np.sqrt((wariancja + wariancja) / 2)

    assert table.loc[0, "SMD po"] == pytest.approx(oczekiwany)
    assert table.loc[0, "SMD przed"] == pytest.approx(
        abs(3.0 - 4.0) / np.sqrt((wariancja + wariancja) / 2)
    )


# --------------------------------------------------------------------------
# 3. Integralnosc par
# --------------------------------------------------------------------------
def test_liczba_par_zgadza_sie_z_macierza_indeksow(sample, main_scenario):
    """Caliper odrzuca pare w calosci: razem z obserwacja i jej sasiadami.

    Gdyby maska odrzucajaca byla nalozona na grupe eksperymentalna, a nie na
    macierz indeksow, obie struktury rozjechalyby sie o liczbe odrzuconych par,
    a roznice parowe laczylyby lokal z sobowtorem innego lokalu.

    Indeksy adresuja ramke kontrolna, nie cala probe modelowa - stad warunek
    na ich zakres. Pomylenie ramek nie zglosi bledu, tylko po cichu podstawi
    lokal przy metrze w roli sobowtora.
    """
    matched = main_scenario["matched"]
    pairs, neighbours = matched["indices"].shape

    assert pairs == len(matched["treated"])
    assert neighbours == matched["neighbours"]
    assert len(psm.paired_differences(matched, config.LOG_OUTCOME)) == pairs
    assert main_scenario["pary"] == pairs
    assert matched["indices"].min() >= 0
    assert matched["indices"].max() < len(matched["control"])
    assert (matched["control"][config.TREATMENT] == 0).all()
    assert pairs + matched["dropped"] == sample["counts"]["grupa_eksperymentalna"]


def test_caliper_ogranicza_najdalszego_sasiada(sample, main_scenario):
    """Limit dotyczy NAJDALSZEGO przyjetego sasiada, nie najblizszego.

    Przy czterech sasiadach sprawdzenie samego pierwszego przepusciloby pary,
    w ktorych trzy pozostale obserwacje sa dowolnie odlegle - a to ich srednia
    wchodzi do roznicy parowej.
    """
    matched = main_scenario["matched"]
    scores = main_scenario["scores"]

    assert matched["caliper_limit"] == pytest.approx(
        config.CALIPER_SD * np.std(scores)
    )
    control_scores = matched["control"]["pscore"].to_numpy()
    treated_scores = matched["treated"]["pscore"].to_numpy()
    odleglosci = np.abs(
        control_scores[matched["indices"]] - treated_scores[:, None]
    )
    assert odleglosci.max() <= matched["caliper_limit"] + 1e-12
    assert matched["dropped"] > 0, "Caliper nie odrzucil zadnej pary."


# --------------------------------------------------------------------------
# 4. Zgodnosc efektu procentowego z ATT na logarytmach
# --------------------------------------------------------------------------
def test_efekt_procentowy_zgodny_z_att_na_logarytmach(main_scenario):
    """Efekt procentowy pochodzi z ATT na logarytmach, nie z ATT w zlotych.

    Wartosc jest przeliczana w tescie niezaleznie od funkcji estymujacej:
    z cen lokali dopasowanych i macierzy indeksow, bez siegania po zwrocone
    oszacowanie. Test zawiedzie, gdyby ktos policzyl efekt jako iloraz ATT
    w zlotych przez srednia cene albo pominal przeksztalcenie wykladnicze.
    """
    matched = main_scenario["matched"]
    treated_log = matched["treated"][config.LOG_OUTCOME].to_numpy()
    neighbours_log = (
        matched["control"]
        .iloc[matched["indices"].flatten()][config.LOG_OUTCOME]
        .to_numpy()
        .reshape(-1, matched["neighbours"])
        .mean(axis=1)
    )
    att_log = (treated_log - neighbours_log).mean()
    oczekiwany = (np.exp(att_log) - 1) * 100

    assert main_scenario["att_log"] == pytest.approx(att_log)
    assert main_scenario["efekt_procentowy"] == pytest.approx(oczekiwany)


def test_sobowtor_usredniany_jest_w_obrebie_pary():
    """Srednia z sasiadow liczona jest osobno dla kazdej pary.

    Usrednienie calej splaszczonej listy daloby te sama wartosc ATT, ale
    jednakowe roznice parowe, a wiec zerowy rozrzut - i w slad za nim zanizona
    wartosc p oraz przedzial ufnosci bez pokrycia w danych. Pomylona kolejnosc
    przy skladaniu macierzy laczylaby z kolei lokal z sasiadami innego lokalu.
    Oba bledy zmieniaja wylacznie roznice parowe, wiec tylko one sa tu
    sprawdzane.
    """
    matched = _synthetic_matched(
        treated_values=np.array([0.0, 0.0]),
        control_values=np.array([1.0, 3.0, 10.0, 20.0]),
        indices=[[0, 1], [2, 3]],
        neighbours=2,
    )

    roznice = psm.paired_differences(matched, config.LOG_OUTCOME)
    np.testing.assert_allclose(roznice, [-2.0, -15.0])

    wynik = psm.estimate(matched)
    assert wynik["att_log"] == pytest.approx(-8.5)
    assert wynik["blad_standardowy_log"] > 0
    assert wynik["pary"] == 2


def test_przedzial_ufnosci_liczony_z_roznic_parowych(main_scenario):
    """Przedzial ufnosci opisuje ten sam estymator co test istotnosci.

    Blad standardowy pochodzi z odchylenia roznic parowych, nie z wariancji
    dwoch niezaleznych prob o roznej licznosci. Przy dopasowaniu 1-do-4 proba
    kontrolna ma czterokrotnie wiecej obserwacji niz jest par, wiec blad
    liczony na probach niezaleznych bylby wezszy bez pokrycia w danych.
    """
    matched = main_scenario["matched"]
    roznice = psm.paired_differences(matched, config.LOG_OUTCOME)
    blad = roznice.std(ddof=1) / np.sqrt(len(roznice))

    assert main_scenario["blad_standardowy_log"] == pytest.approx(blad)
    polowa = config.CONFIDENCE_Z * blad
    assert main_scenario["ci_dolny_procent"] == pytest.approx(
        (np.exp(main_scenario["att_log"] - polowa) - 1) * 100
    )
    assert main_scenario["ci_gorny_procent"] == pytest.approx(
        (np.exp(main_scenario["att_log"] + polowa) - 1) * 100
    )


def test_podgrupy_sumuja_sie_do_proby_dopasowanej(main_scenario):
    """Podzial na podgrupy nie gubi ani nie dubluje zadnej pary."""
    subgroups = psm.subgroup_effects(main_scenario["matched"])
    assert subgroups["N"].sum() == main_scenario["pary"]
