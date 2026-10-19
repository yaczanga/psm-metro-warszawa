"""Sciezki, parametry badania i wspolrzedne stacji metra.

Jedyne miejsce, w ktorym zapisane sa sciezki i stale. Kod poza tym modulem
nie buduje sciezek z literalow.
"""

from pathlib import Path

# Korzen projektu wyznaczony wzgledem polozenia tego pliku: src/config.py ->
# katalog nadrzedny src/ jest korzeniem. Dziala niezaleznie od katalogu
# roboczego, z ktorego wystartowal notebook albo pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

APARTMENTS_CSV = DATA_RAW / "apartments_pl_2024_06.csv"

# --------------------------------------------------------------------------
# Proba
# --------------------------------------------------------------------------
CITY = "warszawa"

# Zabezpieczenie przed cenami, ktore nie podlegaja standardowym mechanizmom
# rynkowym. W tym pliku zadna warszawska oferta nie wypada poza ten przedzial,
# wiec filtr nie usuwa obserwacji - zostaje jako warunek na wypadek podmiany
# danych na inny miesiac.
PRICE_MIN = 150_000
PRICE_MAX = 5_000_000

# --------------------------------------------------------------------------
# Strefa buforowa i zmienna interwencyjna
# --------------------------------------------------------------------------
# Close_to_Metro = 1 dla lokali blizej niz CLOSE_KM od najblizszej stacji,
# 0 dla oddalonych o wiecej niz FAR_KM. Przedzial miedzy progami jest strefa
# buforowa i wypada z proby: wplyw metra jest tam umiarkowany, a przypisanie
# obserwacji do ktorejkolwiek grupy byloby arbitralne.
CLOSE_KM = 0.5
FAR_KM = 1.2

# Przeliczniki stopnia na kilometry dla szerokosci Warszawy. Odleglosc liczona
# jest euklidesowo w tej lokalnej metryce, nie po trasie dojscia.
LAT_KM = 111.1
LON_KM = 68.2

# --------------------------------------------------------------------------
# Zmienne modelu
# --------------------------------------------------------------------------
OUTCOME = "Price_per_m2"
LOG_OUTCOME = "Log_Price"
TREATMENT = "Close_to_Metro"
DISTANCE = "Distance_to_Metro"

NUMERIC_COLUMNS = [
    "squareMeters",
    "buildYear",
    "rooms",
    "centreDistance",
    "floorCount",
    "poiCount",
    "schoolDistance",
    "clinicDistance",
    "pharmacyDistance",
    "floor",
    "postOfficeDistance",
    "kindergartenDistance",
    "restaurantDistance",
    "collegeDistance",
]

BINARY_COLUMNS = [
    "hasParkingSpace",
    "hasBalcony",
    "hasElevator",
    "hasSecurity",
    "hasStorageRoom",
]

CATEGORICAL_COLUMNS = ["ownership", "buildingMaterial", "type"]

# Kolumny techniczne i wynikowe, ktore nie moga trafic do wektora cech.
EXCLUDED_FROM_FEATURES = [
    "id",
    "city",
    "latitude",
    "longitude",
    "price",
    "condition",
    OUTCOME,
    LOG_OUTCOME,
    TREATMENT,
    DISTANCE,
]

# Zmienna rozdzielajaca podproby w analizie heterogenicznosci.
MATERIAL_COLUMN = "buildingMaterial_concreteSlab"

# --------------------------------------------------------------------------
# Warianty dopasowania
# --------------------------------------------------------------------------
# Caliper wyrazony jest w odchyleniach standardowych SAMEGO propensity score,
# nie jego logitu. Konwencja Austina (2011) zaklada logit, wiec prog liczbowy
# nie jest z nia porownywalny.
CALIPER_SD = 0.2

SCENARIOS = [
    {"name": "Logit 1-to-1", "model": "logit", "neighbours": 1, "caliper_sd": None},
    {"name": "Probit 1-to-1", "model": "probit", "neighbours": 1, "caliper_sd": None},
    {"name": "Logit 1-to-4", "model": "logit", "neighbours": 4, "caliper_sd": None},
    {"name": "Probit 1-to-4", "model": "probit", "neighbours": 4, "caliper_sd": None},
    {"name": "Logit 1-to-4 Caliper", "model": "logit", "neighbours": 4,
     "caliper_sd": CALIPER_SD},
    {"name": "Probit 1-to-4 Caliper", "model": "probit", "neighbours": 4,
     "caliper_sd": CALIPER_SD},
]

# Wariant wybrany do interpretacji i do rysunkow diagnostycznych.
MAIN_SCENARIO = "Logit 1-to-4 Caliper"

# Parametry estymacji propensity score. Odwrotnosc sily regularyzacji ustawiona
# jest bardzo wysoko, bo celem jest oszacowanie bez obciazenia, nie prognoza.
LOGIT_MAX_ITER = 5000
LOGIT_C = 1e9
PROBIT_MAX_ITER = 200

# --------------------------------------------------------------------------
# Progi diagnostyczne
# --------------------------------------------------------------------------
# Standaryzowana roznica srednich (SMD) wedlug Rubina (2001): 0,1 to poziom
# docelowy, 0,25 to gorna granica akceptowalnego zbalansowania.
SMD_TARGET = 0.1
SMD_LIMIT = 0.25

# Prog, powyzej ktorego wspolliniowosc uznaje sie za problem.
VIF_LIMIT = 10.0

CONFIDENCE_Z = 1.96

# --------------------------------------------------------------------------
# Stacje metra M1 i M2 - wspolrzedne geograficzne
# --------------------------------------------------------------------------
METRO_STATIONS = {
    "Ratusz-Arsenał": (52.2452163, 21.0008823),
    "Marymont": (52.2715768, 20.9719399),
    "Stare Bielany": (52.2818277, 20.9493511),
    "Wawrzyszew": (52.2863474, 20.9395150),
    "Młociny": (52.2907703, 20.9298678),
    "Słodowiec": (52.2768261, 20.9601259),
    "Dworzec Wileński": (52.2537771, 21.0357972),
    "Stadion Narodowy": (52.2468346, 21.0428470),
    "Centrum Nauki Kopernik": (52.2399148, 21.0317878),
    "Nowy Świat-Uniwersytet": (52.2368196, 21.0168168),
    "Rondo ONZ": (52.2330735, 20.9981024),
    "Rondo Daszyńskiego": (52.2300827, 20.9828946),
    "Kabaty": (52.1320765, 21.0650711),
    "Stokłosy": (52.1560759, 21.0347233),
    "Natolin": (52.1411007, 21.0564351),
    "Imielin": (52.1493000, 21.0461062),
    "Ursynów": (52.1620456, 21.0276283),
    "Służew": (52.1727624, 21.0262866),
    "Racławicka": (52.1988637, 21.0122349),
    "Wierzbno": (52.1898719, 21.0167966),
    "Wilanowska": (52.1818168, 21.0231452),
    "Pole Mokotowskie": (52.2087775, 21.0079298),
    "Politechnika": (52.2186581, 21.0153031),
    "Centrum": (52.2310069, 21.0101860),
    "Dworzec Gdański": (52.2580586, 20.9941857),
    "Plac Wilsona": (52.2692619, 20.9844973),
    "Płocka": (52.2324542, 20.9663847),
    "Trocka": (52.2751021, 21.0550586),
    "Świętokrzyska": (52.2350954, 21.0078988),
    "Księcia Janusza": (52.2391822, 20.9443771),
    "Młynów": (52.2376624, 20.9601048),
    "Szwedzka": (52.2634709, 21.0455232),
    "Targówek Mieszkaniowy": (52.2692518, 21.0513658),
    "Bemowo": (52.2392071, 20.9154991),
    "Ulrychów": (52.2403314, 20.9298652),
    "Bródno": (52.2935850, 21.0289387),
    "Kondratowicza": (52.2920848, 21.0486889),
    "Zacisze": (52.2837496, 21.0621480),
}


def ensure_output_dirs() -> None:
    """Tworzy katalogi wynikowe, jesli jeszcze nie istnieja."""
    for path in (FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)
