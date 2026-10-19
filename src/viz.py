"""Rysunki projektu.

Barwy pochodza wylacznie z modulu plot_style i sa wybierane przez role, jaka
seria pelni w danych: cynober to przedmiot uwagi, blekit to odniesienie.
Grupa eksperymentalna - lokale przy metrze - jest przedmiotem badania, wiec
nosi cynober na kazdym rysunku zestawu; grupa kontrolna jest odniesieniem.

Wykresy rysuje sam matplotlib. Biblioteki rysujace wlasne figury dobieraja
barwy po swojemu, a tu jedynym ich zrodlem ma byc plot_style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config, plot_style as style

style.apply_style()

# Spacja nierozdzielajaca jako separator tysiecy, zgodnie z zapisem liczb
# w pozostalych napisach projektu.
NBSP = " "

TREATED_LABEL = "blisko metra"
CONTROL_LABEL = "daleko od metra"


def _save(fig, name: str, save: bool) -> None:
    """Zapisuje rysunek do katalogu wynikowego pod podana nazwa."""
    if not save:
        return
    config.ensure_output_dirs()
    fig.savefig(config.FIGURES / name, dpi=style.SAVE_DPI, bbox_inches="tight")


def _thousands_axis(ax, axis: str = "y") -> None:
    """Separator tysiecy jako spacja nierozdzielajaca."""
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", NBSP))
    if "x" in axis:
        ax.xaxis.set_major_formatter(formatter)
    if "y" in axis:
        ax.yaxis.set_major_formatter(formatter)


def plot_buffer_zone(model_frame: pd.DataFrame, save: bool = False):
    """Rozklad odleglosci do metra z zaznaczona strefa buforowa.

    Rysunek pokazuje, na czym stoi identyfikacja efektu: do proby wchodza
    wylacznie lokale wyraznie blisko albo wyraznie daleko od stacji, a pas
    miedzy progami wypada. Odleglosci sa jedna seria, wiec nosza barwe
    odniesienia; akcent nalezy do wykluczonego pasa, bo to on jest decyzja
    opisywana przez ten rysunek.
    """
    distances = model_frame[config.DISTANCE]
    fig, ax = plt.subplots(figsize=style.STANDARD)

    # Krok kosza dobrany tak, zeby oba progi strefy buforowej wypadly dokladnie
    # na krawedziach: kosz zachodzacy na prog wystawalby w zacieniony pas
    # i sugerowal obserwacje, ktorych w probie nie ma.
    upper = min(float(distances.max()), 8.0)
    bins = np.arange(0, upper + 0.1, 0.1)
    ax.hist(distances.clip(upper=upper), bins=bins, color=style.MUTED,
            edgecolor=style.BACKGROUND, linewidth=0.4)

    ax.axvspan(config.CLOSE_KM, config.FAR_KM, color=style.ACCENT, alpha=0.18,
               zorder=0)
    for threshold in (config.CLOSE_KM, config.FAR_KM):
        ax.axvline(threshold, color=style.ACCENT, linewidth=1.2, linestyle="--")

    ax.set_title("Odległość do najbliższej stacji metra")
    ax.set_xlabel("Distance_to_Metro (km)")
    ax.set_ylabel("liczba ofert")
    ax.annotate(
        "strefa buforowa\n0,5-1,2 km, wykluczona",
        xy=((config.CLOSE_KM + config.FAR_KM) / 2, ax.get_ylim()[1] * 0.92),
        xytext=(config.FAR_KM + 0.9, ax.get_ylim()[1] * 0.92),
        color=style.TEXT_SECONDARY, fontsize=10, va="center",
        arrowprops=dict(arrowstyle="-", color=style.ACCENT, linewidth=1.0),
    )
    style.decimal_comma(ax, axis="x", places=1)
    _thousands_axis(ax, axis="y")

    fig.tight_layout()
    _save(fig, "01_strefa_buforowa.png", save)
    return fig


def plot_propensity_overlap(matched: dict, scores, treatment: pd.Series,
                            save: bool = False):
    """Rozklad propensity score przed i po dopasowaniu, w dwoch panelach.

    Sa tu trzy zbiory obserwacji - grupa eksperymentalna, cala grupa kontrolna
    i jej czesc dopasowana - ale tylko dwa byty: grupa badana i grupa kontrolna
    w dwoch stanach. Podzial na panele oddaje to wprost i utrzymuje dwie barwy
    tozsamosci; trzecia barwa w jednym panelu sugerowalaby trzecia grupe.
    """
    scores = np.asarray(scores)
    treated_scores = scores[treatment.to_numpy() == 1]
    control_scores = scores[treatment.to_numpy() == 0]
    matched_control = matched["control"].iloc[
        matched["indices"].flatten()
    ]["pscore"].to_numpy()
    matched_treated = matched["treated"]["pscore"].to_numpy()

    grid = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=style.WIDE, sharey=True)

    panels = (
        (axes[0], "Przed dopasowaniem", treated_scores, control_scores),
        (axes[1], "Po dopasowaniu", matched_treated, matched_control),
    )
    for ax, title, treated_values, control_values in panels:
        for values, color, label in (
            (control_values, style.REFERENCE, CONTROL_LABEL),
            (treated_values, style.ACCENT, TREATED_LABEL),
        ):
            density = stats.gaussian_kde(values)(grid)
            ax.fill_between(grid, density, color=color, alpha=0.25)
            ax.plot(grid, density, color=color, linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel("propensity score")
        style.decimal_comma(ax, axis="x", places=1)
        style.decimal_comma(ax, axis="y", places=1)

    axes[0].set_ylabel("gęstość")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "02_rozklad_propensity_score.png", save)
    return fig


def plot_love(smd: pd.DataFrame, save: bool = False):
    """Wykres Love'a: standaryzowana roznica srednich przed i po dopasowaniu.

    Stan przed dopasowaniem jest blad selekcji, czyli zdarzenie negatywne, wiec
    nosi barwe ryzyka; stan po dopasowaniu jest stanem docelowym i nosi barwe
    odniesienia. Progi Rubina sa opisane na rysunku, a nie w legendzie - sa
    anotacja, nie seria danych.
    """
    table = smd.sort_values("SMD przed", ascending=True)
    positions = np.arange(len(table))

    fig, ax = plt.subplots(figsize=style.SQUARE)
    ax.hlines(positions, table["SMD po"], table["SMD przed"],
              color=style.GRID, linewidth=1.2, zorder=1)
    ax.scatter(table["SMD przed"], positions, color=style.RISK, s=55,
               zorder=3, label="przed dopasowaniem")
    ax.scatter(table["SMD po"], positions, color=style.REFERENCE, s=55,
               zorder=3, label="po dopasowaniu")

    # Podpisy progow rozsuniete w pionie: przy 0,1 i 0,25 na jednej wysokosci
    # napisy nachodza na siebie.
    thresholds = (
        (config.SMD_TARGET, "0,1 - poziom docelowy", len(table) - 0.1),
        (config.SMD_LIMIT, "0,25 - granica Rubina", len(table) - 1.3),
    )
    for threshold, text, height in thresholds:
        ax.axvline(threshold, color=style.TEXT_SECONDARY, linewidth=1.0,
                   linestyle="--", zorder=2)
        ax.annotate(text, xy=(threshold + 0.012, height),
                    color=style.TEXT_SECONDARY, fontsize=9, va="center")

    ax.set_yticks(positions)
    ax.set_yticklabels(table["Zmienna"], fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.8, len(table) + 0.6)
    ax.set_title("Zbalansowanie zmiennych kontrolnych")
    ax.set_xlabel("standaryzowana różnica średnich (SMD)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    style.decimal_comma(ax, axis="x", places=2)

    fig.tight_layout()
    _save(fig, "03_wykres_lowea.png", save)
    return fig


def plot_scenarios(table: pd.DataFrame, save: bool = False):
    """Efekt procentowy w kazdym wariancie wraz z 95% przedzialem ufnosci.

    Jedna seria oszacowan, w ktorej jeden element jest wyrozniony: wariant
    wybrany do interpretacji nosi akcent, pozostale sa tlem. To wzorzec akcentu
    na tle masy, nie szesc kategorii o wlasnych barwach.
    """
    frame = table.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(frame))
    is_main = frame["Wariant"] == config.MAIN_SCENARIO

    fig, ax = plt.subplots(figsize=style.STANDARD)
    ax.axvline(0, color=style.AXES, linewidth=1.0, zorder=1)

    for position in positions:
        color = style.ACCENT if is_main.iloc[position] else style.MUTED
        ax.hlines(position, frame["95% CI dolny (%)"].iloc[position],
                  frame["95% CI górny (%)"].iloc[position], color=color,
                  linewidth=2.2, zorder=2)
        ax.plot(frame["ATT (%)"].iloc[position], position, "o", color=color,
                markersize=7, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels(frame["Wariant"], fontsize=10)
    ax.set_title("Oszacowania ATT w wariantach dopasowania")
    ax.set_xlabel("efekt procentowy z 95% przedziałem ufności (%)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.decimal_comma(ax, axis="x", places=1)

    # Seria jest jedna, wiec legendy nie ma: wyrozniony wariant nosi akcent
    # takze w etykiecie osi, a opis stoi przy samym oszacowaniu.
    main_position = int(np.flatnonzero(is_main.to_numpy())[0])
    ax.get_yticklabels()[main_position].set_color(style.ACCENT)
    ax.annotate(
        "wariant wybrany do interpretacji",
        xy=(frame["95% CI dolny (%)"].iloc[main_position] - 0.2, main_position),
        color=style.ACCENT, fontsize=10, va="center", ha="right",
    )

    fig.tight_layout()
    _save(fig, "04_oszacowania_wariantow.png", save)
    return fig


def plot_material_subgroups(subgroups: pd.DataFrame, save: bool = False):
    """Efekt procentowy w podgrupach wyodrebnionych materialem budynku.

    Dwie kategorie nominalne, wiec barwy tozsamosci. Wielka plyta jest
    przedmiotem uwagi - to w tej podgrupie efekt zmienia znak - a cegla
    i nowe budownictwo sa odniesieniem jako podgrupa liczniejsza.
    """
    frame = subgroups.copy()
    colors = [
        style.ACCENT if label.startswith("wielka") else style.REFERENCE
        for label in frame["Podgrupa"]
    ]
    labels = [
        f"{row.Podgrupa}\n(N = {row.N})" for row in frame.itertuples()
    ]

    fig, ax = plt.subplots(figsize=style.SMALL)
    ax.bar(labels, frame["Efekt (%)"], color=colors, width=0.55)
    ax.axhline(0, color=style.AXES, linewidth=1.0)
    ax.set_title("Premia za bliskość metra według materiału budynku")
    ax.set_ylabel("efekt procentowy (%)")
    style.decimal_comma(ax, axis="y", places=0)
    style.value_labels(ax, places=2, suffix="%")

    margin = max(abs(frame["Efekt (%)"])) * 0.35
    ax.set_ylim(min(frame["Efekt (%)"]) - margin, max(frame["Efekt (%)"]) + margin)

    fig.tight_layout()
    _save(fig, "05_premia_wedlug_materialu.png", save)
    return fig


def plot_vif(vif: pd.DataFrame, save: bool = False):
    """Czynnik inflacji wariancji zmiennych kontrolnych.

    Jedna seria, w ktorej wyrozniona jest zmienna o najwyzszej wartosci -
    akcent na tle masy. Prog jest anotacja, nie seria.
    """
    table = vif.sort_values("VIF", ascending=True)
    positions = np.arange(len(table))
    highest = table["VIF"].idxmax()
    colors = [
        style.ACCENT if index == highest else style.MUTED
        for index in table.index
    ]

    fig, ax = plt.subplots(figsize=style.SQUARE)
    ax.barh(positions, table["VIF"], color=colors, height=0.68)
    ax.axvline(config.VIF_LIMIT, color=style.TEXT_SECONDARY, linewidth=1.0,
               linestyle="--")
    # Podpis progu po lewej stronie linii: przy progu rownym krancowi osi napis
    # postawiony na prawo wychodzi poza obszar rysunku.
    ax.annotate(f"próg {config.VIF_LIMIT:.0f}",
                xy=(config.VIF_LIMIT - 0.2, len(table) - 0.6),
                color=style.TEXT_SECONDARY, fontsize=9, va="center",
                ha="right")

    ax.set_yticks(positions)
    ax.set_yticklabels(table["Zmienna"], fontsize=9)
    ax.set_ylim(-0.8, len(table) + 0.4)
    ax.set_title("Współliniowość zmiennych kontrolnych")
    ax.set_xlabel("czynnik inflacji wariancji (VIF)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.decimal_comma(ax, axis="x", places=0)

    fig.tight_layout()
    _save(fig, "06_vif_zmiennych_kontrolnych.png", save)
    return fig


def plot_correlation(matrix: pd.DataFrame, save: bool = False):
    """Macierz korelacji zmiennych kontrolnych.

    Korelacja jest wielkoscia biegunowa wokol zera, wiec nosi rampe
    dywergencyjna. Przekatna i gorny trojkat sa zamaskowane, pusty pierwszy
    wiersz i ostatnia kolumna wypadaja z rysunku, a komorki dostaja
    obramowanie, bo wypelnienie bliskie zeru znika na tle.
    """
    trimmed = matrix.iloc[1:, :-1]
    mask = style.upper_triangle_mask(matrix)[1:, :-1]
    values = np.ma.masked_array(trimmed.to_numpy(), mask=mask)

    fig, ax = plt.subplots(figsize=style.SQUARE)
    image = ax.pcolormesh(values, cmap=style.DIVERGING_CMAP, vmin=-1, vmax=1,
                          edgecolors=style.GRID, linewidth=0.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks(np.arange(len(trimmed.columns)) + 0.5)
    ax.set_yticks(np.arange(len(trimmed.index)) + 0.5)
    ax.set_xticklabels(trimmed.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(trimmed.index, fontsize=8)
    ax.set_title("Korelacje między zmiennymi kontrolnymi")
    ax.grid(visible=False)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.outline.set_visible(False)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    bar.set_ticklabels(["-1,0", "-0,5", "0,0", "0,5", "1,0"])
    bar.ax.tick_params(labelsize=9, colors=style.TEXT_SECONDARY)

    fig.tight_layout()
    _save(fig, "07_macierz_korelacji.png", save)
    return fig
