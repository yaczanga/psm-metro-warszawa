"""Wczytanie ogloszen, odleglosc do metra i przygotowanie proby modelowej."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_listings(path=None) -> pd.DataFrame:
    """Wczytuje plik ogloszen i zwraca wszystkie miasta bez zmian."""
    path = config.APARTMENTS_CSV if path is None else path
    frame = pd.read_csv(path)
    required = ("city", "price", "squareMeters", "latitude", "longitude")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Brak kolumn w {path}: {missing}")
    return frame


def select_city(frame: pd.DataFrame, city=None) -> pd.DataFrame:
    """Zostawia ogloszenia z jednego miasta, niezaleznie od wielkosci liter."""
    city = config.CITY if city is None else city
    return frame[frame["city"].str.lower() == city].copy()


def apply_price_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Odrzuca oferty poza przedzialem cen uznanym za rynkowy."""
    return frame[
        (frame["price"] > config.PRICE_MIN) & (frame["price"] < config.PRICE_MAX)
    ].copy()


def distance_to_metro(latitude: float, longitude: float) -> float:
    """Odleglosc euklidesowa w km do najblizszej stacji metra.

    Stopnie przeliczane sa na kilometry lokalnymi wspolczynnikami dla
    szerokosci Warszawy. Wynikiem jest odleglosc w linii prostej, nie dlugosc
    trasy dojscia - dla lokalu po drugiej stronie torow albo rzeki rzeczywista
    droga jest dluzsza.
    """
    best = np.inf
    for station_lat, station_lon in config.METRO_STATIONS.values():
        delta_lat = (latitude - station_lat) * config.LAT_KM
        delta_lon = (longitude - station_lon) * config.LON_KM
        distance = np.sqrt(delta_lat**2 + delta_lon**2)
        if distance < best:
            best = distance
    return float(best)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cene za metr, jej logarytm i odleglosc do najblizszej stacji."""
    frame = frame.copy()
    frame[config.OUTCOME] = frame["price"] / frame["squareMeters"]
    frame[config.LOG_OUTCOME] = np.log(frame[config.OUTCOME])
    frame[config.DISTANCE] = [
        distance_to_metro(lat, lon)
        for lat, lon in zip(frame["latitude"], frame["longitude"])
    ]
    return frame


def missing_share(frame: pd.DataFrame) -> pd.Series:
    """Udzial brakow w kazdej zmiennej modelu, w procentach, malejaco.

    Wartosci opisuja skale usuwania listowego: zmienna z najwyzszym udzialem
    decyduje o tym, ile proby ubywa.
    """
    columns = (
        config.NUMERIC_COLUMNS + config.BINARY_COLUMNS + config.CATEGORICAL_COLUMNS
    )
    share = frame[columns].isna().mean() * 100
    return share.sort_values(ascending=False)


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Usuwa listowo braki w zmiennych modelu i koduje je liczbowo.

    Braki usuwane sa listowo, bo algorytm najblizszego sasiada liczy odleglosc
    na pelnym wektorze cech. Zmienne tak-nie przechodza na 0 i 1, zmienne
    kategorialne na zbior zmiennych zero-jedynkowych z pominieciem pierwszego
    poziomu, ktory zostaje poziomem bazowym.
    """
    columns = (
        config.NUMERIC_COLUMNS + config.BINARY_COLUMNS + config.CATEGORICAL_COLUMNS
    )
    frame = frame.dropna(subset=columns).copy()
    for column in config.BINARY_COLUMNS:
        frame[column] = (frame[column] == "yes").astype(int)
    return pd.get_dummies(
        frame, columns=config.CATEGORICAL_COLUMNS, drop_first=True, dtype=int
    )


def apply_buffer_zone(frame: pd.DataFrame) -> pd.DataFrame:
    """Odrzuca strefe buforowa i nadaje zmienna interwencyjna.

    Zostaja wylacznie lokale blizej niz CLOSE_KM albo dalej niz FAR_KM od
    stacji. Obserwacje z przedzialu miedzy progami wypadaja z proby.
    """
    frame = frame[
        (frame[config.DISTANCE] < config.CLOSE_KM)
        | (frame[config.DISTANCE] > config.FAR_KM)
    ].copy()
    frame[config.TREATMENT] = (frame[config.DISTANCE] < config.CLOSE_KM).astype(int)
    return frame


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Macierz zmiennych kontrolnych: wszystko poza kolumnami technicznymi.

    Zmienna wynikowa, interwencyjna i sama odleglosc do metra sa wykluczone -
    propensity score ma opisywac cechy fizyczne lokalu, nie jego polozenie
    wobec stacji.
    """
    columns = [c for c in frame.columns if c not in config.EXCLUDED_FROM_FEATURES]
    return frame[columns].select_dtypes(include=[np.number]).dropna()


def build_model_sample(path=None) -> dict:
    """Przeprowadza caly tor przygotowania danych i zwraca probe modelowa.

    Zwraca slownik z ramka proby, macierza cech, zmienna interwencyjna oraz
    licznosciami kolejnych krokow - te ostatnie sa wejsciem do opisu proby
    i do testow, zeby zadna licznosc nie byla wpisana w kod na sztywno.
    """
    listings = load_listings(path)
    city = select_city(listings)
    priced = apply_price_filter(city)
    derived = add_derived_columns(priced)
    missing = missing_share(derived)
    encoded = encode(derived)
    model = apply_buffer_zone(encoded)

    features = feature_matrix(model)
    treatment = model.loc[features.index, config.TREATMENT]
    model = model.loc[features.index]

    counts = {
        "oferty_w_pliku": len(listings),
        "oferty_w_miescie": len(city),
        "stacje_metra": len(config.METRO_STATIONS),
        "po_filtrze_cen": len(priced),
        "po_usunieciu_brakow": len(encoded),
        "proba_modelowa": len(model),
        "grupa_eksperymentalna": int(treatment.sum()),
        "grupa_kontrolna": int((1 - treatment).sum()),
        "zmienne_kontrolne": features.shape[1],
    }
    counts["utracony_udzial"] = round(
        100 * (1 - counts["po_usunieciu_brakow"] / counts["oferty_w_miescie"]), 1
    )

    return {
        "model": model,
        "features": features,
        "treatment": treatment,
        "counts": counts,
        "missing_share": missing,
    }


def raw_difference(model: pd.DataFrame) -> float:
    """Roznica srednich cen za metr miedzy grupami przed dopasowaniem."""
    close = model[model[config.TREATMENT] == 1][config.OUTCOME]
    far = model[model[config.TREATMENT] == 0][config.OUTCOME]
    return float(close.mean() - far.mean())
