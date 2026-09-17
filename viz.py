"""
viz.py - Visualization layer for the Climate Risk Explorer (point mode)
==========================================================================
All matplotlib/cartopy rendering and indicator presentation metadata (which
category each indicator belongs to, which colormap suits it) lives here,
kept separate from backend.py (data) and app.py (Streamlit orchestration).
Nothing here imports streamlit.

Indicator categories and colours
---------------------------------
The workbook's Metadata sheet gives each indicator a full name/definition/
units, but not a category or a suggested colour - that's presentation
judgement, not data, so it's encoded here rather than invented at the data
layer. Every indicator maps to a category (for a clean two-level picker) and
a perceptually-uniform sequential colormap chosen for what it physically
represents (wetness -> blue family, dryness -> brown/orange, heat -> red/
orange, cold -> blue/purple family, energy demand indicators tinted by
which season drives them). An indicator not in this dict (e.g. a future
addition to the workbook) still works, falling back to a neutral default
and an "Other" category rather than failing.
"""
from __future__ import annotations

import socket
import contextlib
import textwrap
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

import backend as be   # scenario identity only (no plotting in backend, no cycle)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

# cartopy's feature downloader has no timeout of its own - on a firewalled or
# slow connection, fetching Natural Earth data on first use can hang
# indefinitely. IMPORTANT: this must be scoped to just the download calls via
# the context manager below, NOT set globally with socket.setdefaulttimeout()
# at import time - a process-wide default timeout also throttles every other
# socket the host application opens, including the framework's own live
# connection to the browser, which silently breaks the whole app (observed:
# a permanently blank page with no Python exception, since nothing crashes -
# a long-lived connection just never completes within the global timeout).
CARTOPY_TIMEOUT_SEC = 8


@contextlib.contextmanager
def _scoped_socket_timeout(seconds: float):
    """Temporarily set the default socket timeout, restoring whatever it was
    before on exit - so this only affects sockets opened inside the `with`
    block (cartopy's feature download), not the rest of the process."""
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)

# (category, colormap) per indicator variable name
INDICATOR_PRESENTATION = {
    # --- Precipitation ---
    "rain_days":            ("Precipitation", "Blues"),
    "dry_days":              ("Precipitation", "YlOrBr"),
    "max_consec_dry_days":   ("Precipitation", "YlOrBr"),
    "rx1day":                ("Precipitation", "GnBu"),
    "rx5day":                ("Precipitation", "GnBu"),
    "very_heavy_rain_days":  ("Precipitation", "PuBu"),
    # --- Temperature extremes ---
    "hot_days":              ("Temperature Extremes", "YlOrRd"),
    "warm_nights":           ("Temperature Extremes", "OrRd"),
    "cold_days":             ("Temperature Extremes", "Blues"),
    "cold_nights":           ("Temperature Extremes", "PuBu"),
    "frost_days":            ("Temperature Extremes", "BuPu"),
    "dtr":                   ("Temperature Extremes", "viridis"),
    # --- Degree days (energy demand) ---
    "cooling_degree_days":   ("Degree Days", "YlOrRd"),
    "heating_degree_days":   ("Degree Days", "Blues"),
    # --- Heat stress (Mar-Jun) ---
    "heatwave_frequency_summer":     ("Heat Stress (Mar-Jun)", "YlOrRd"),
    "heat_index_annual_mean_summer": ("Heat Stress (Mar-Jun)", "inferno"),
    "heat_index_valid_days_summer":  ("Heat Stress (Mar-Jun)", "viridis"),
}
DEFAULT_CATEGORY = "Other"
DEFAULT_COLORMAP = "YlOrRd"

# Professional, muted basemap palette (data colours are meant to pop against this)
BASEMAP_STYLE = dict(
    land_color="#f2efe9",
    ocean_color="#e6f0f5",
    coastline_color="#4a4a4a",
    border_color="#8a8a8a",
    gridline_color="#b0b0b0",
)


# --------------------------------------------------------------------------
# Typography
# --------------------------------------------------------------------------
# One place decides every text size in every figure. Each named scale sets a
# base size and derives the rest from it (ticks below base, axis labels at
# base, panel title above it), so the type hierarchy stays proportional
# instead of drifting as individual fontsize= arguments get tweaked - there
# are deliberately no hardcoded fontsize values in the plotting functions
# below.
#
# The figure size shrinks as the base size grows, which looks backwards but
# is not: the host application scales the rendered figure to the width of its
# container, so text size on screen is set by the *ratio* of font size to
# figure width, not by the font size alone. A 16 pt label on a 13 in canvas
# scaled down to fit reads smaller than a 16 pt label on a 9 in canvas. Each
# scale therefore pairs a base size with a canvas width that keeps the result
# legible, and the aspect ratio is kept near 2.6:1 so the two panels stay
# usable.
TEXT_SCALES: dict[str, dict] = {
    "Compact":      dict(base=8.5,  figsize=(13.5, 5.0)),
    "Standard":     dict(base=10.5, figsize=(12.0, 4.7)),
    "Large":        dict(base=13.0, figsize=(10.5, 4.3)),
    "Presentation": dict(base=16.0, figsize=(9.0, 4.0)),
}
DEFAULT_TEXT_SCALE = "Standard"

MUTED_TEXT_COLOR = "#5a5a5a"
GRID_COLOR = "#cccccc"


def apply_typography(scale_name: str = DEFAULT_TEXT_SCALE) -> dict:
    """Set the matplotlib rcParams for the chosen named text scale and return
    its config (base size + figure size). Call once per render, before
    creating the figure - rcParams are read at artist-creation time, so
    changing them afterwards has no effect on artists already made.

    Note: spine visibility is deliberately NOT set through rcParams here.
    `axes.spines.top/right` are applied per named spine at Axes creation, and
    a cartopy GeoAxes has a single spine named 'geo' instead of the four
    rectangular ones - so driving spines from rcParams styles the time-series
    panel and silently does nothing (or worse) on the map panel. The
    rectangular panels hide their own top/right spines in `_style_panel()`.
    """
    cfg = TEXT_SCALES.get(scale_name, TEXT_SCALES[DEFAULT_TEXT_SCALE])
    base = float(cfg["base"])
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": base,
        "axes.titlesize": base + 1.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": base * 0.6,
        "axes.labelsize": base,
        "axes.labelpad": base * 0.4,
        "xtick.labelsize": base - 1.0,
        "ytick.labelsize": base - 1.0,
        "legend.fontsize": base - 1.0,
        "legend.title_fontsize": base - 0.5,
        "figure.titlesize": base + 2.5,
        "figure.titleweight": "bold",
        "axes.axisbelow": True,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": None,       # the app manages layout itself
    })
    return cfg


def text_scale_config(scale_name: str = DEFAULT_TEXT_SCALE) -> dict:
    return TEXT_SCALES.get(scale_name, TEXT_SCALES[DEFAULT_TEXT_SCALE])


def _fs(delta: float = 0.0) -> float:
    """A size relative to the active base font size, for the few annotations
    that have no rcParam of their own (in-panel notes, bar value labels)."""
    return float(mpl.rcParams["font.size"]) + delta


def _wrap(text: str, width: int = 42) -> str:
    """Wrap long indicator names so a bigger text scale widens the title over
    two lines instead of overflowing the panel."""
    return "\n".join(textwrap.wrap(str(text), width=width)) or str(text)


def _style_panel(ax) -> None:
    """Shared styling for a rectangular (non-geographic) panel."""
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        if side in ax.spines:
            ax.spines[side].set_color("#7a7a7a")
    ax.grid(True, axis="both", alpha=0.35, color=GRID_COLOR, linewidth=0.6)


def add_figure_title(fig, text: str) -> None:
    """Figure-level title (location + period), so each panel's own title can
    stay short. Paired with the caller's layout rect, which reserves the top
    strip for it."""
    fig.suptitle(text, x=0.01, ha="left")


# --------------------------------------------------------------------------
# Scenario colours
# --------------------------------------------------------------------------
# The IPCC AR6 scenario colour set (Fig. SPM.8 / AR6 WGI technical colour
# guidance), so a figure from this app is directly comparable with published
# AR6 figures and with anything else drawn to the same convention. Historical
# is near-black, as in AR6, because the observed-forced run is not one of the
# scenario branches. Ordering (see backend.SCENARIO_ORDER) is by radiative
# forcing, not alphabetical, so the legend reads low-forcing to high-forcing.
IPCC_SCENARIO_COLORS = {
    "historical": "#1b1b1b",
    "ssp119":     "#00a9cf",
    "ssp126":     "#003466",
    "ssp245":     "#f69320",
    "ssp370":     "#df0000",
    "ssp434":     "#2274ae",
    "ssp460":     "#b0724e",
    "ssp534os":   "#92397a",
    "ssp585":     "#980002",
}
# For a workbook whose scenario name we do not recognise - qualitatively
# distinct and colour-vision-safe, so it never reads as one of the SSPs.
FALLBACK_SCENARIO_COLORS = ["#117733", "#88ccee", "#cc6677", "#aa4499", "#44aa99"]

# Diverging pair for anomaly bars: warm = above baseline, cool = below.
ANOMALY_COLORS = dict(positive="#b2182b", negative="#2166ac", zero="#999999")


def scenario_color(label: str, fallback_index: int = 0) -> str:
    """Colour for a scenario label, keyed off its canonical identity so
    "ssp245", "SSP2-4.5" and "Projection ssp245 (2015-2037)" all get the same
    colour."""
    key = be.scenario_key(label)
    if key in IPCC_SCENARIO_COLORS:
        return IPCC_SCENARIO_COLORS[key]
    return FALLBACK_SCENARIO_COLORS[fallback_index % len(FALLBACK_SCENARIO_COLORS)]


def scenario_colors(labels: Sequence[str]) -> dict[str, str]:
    """{label: colour} for a set of scenarios, assigning fallback colours in
    order to any unrecognised ones."""
    out, n_unknown = {}, 0
    for label in labels:
        if be.scenario_key(label) in IPCC_SCENARIO_COLORS:
            out[label] = scenario_color(label)
        else:
            out[label] = scenario_color(label, n_unknown)
            n_unknown += 1
    return out


def get_category(indicator: str) -> str:
    return INDICATOR_PRESENTATION.get(indicator, (DEFAULT_CATEGORY, DEFAULT_COLORMAP))[0]


def get_colormap(indicator: str) -> str:
    return INDICATOR_PRESENTATION.get(indicator, (DEFAULT_CATEGORY, DEFAULT_COLORMAP))[1]


def get_colormap_line_color(indicator: str) -> str:
    """A single saturated colour sampled from the indicator's own colormap,
    so the time-series line visually matches the same colour family used for
    that indicator's spatial/legend colouring elsewhere - e.g. a blue line
    for rain_days, a red line for hot_days - rather than every indicator
    defaulting to the same line colour regardless of what it represents."""
    cmap = plt.get_cmap(get_colormap(indicator))
    r, g, b, _ = cmap(0.75)
    return (r, g, b)


def group_indicators_by_category(indicators: list[str], indicator_metadata: dict) -> dict[str, list[str]]:
    """{category: [indicator_var_names...]}, in a fixed, sensible category
    order (falling back to alphabetical for any unrecognised indicator's
    'Other' bucket), with indicators inside each category ordered by their
    full name for a predictable, scannable picker."""
    category_order = ["Precipitation", "Temperature Extremes", "Degree Days", "Heat Stress (Mar-Jun)", DEFAULT_CATEGORY]
    grouped: dict[str, list[str]] = {c: [] for c in category_order}
    for ind in indicators:
        grouped[get_category(ind)].append(ind)
    for cat in grouped:
        grouped[cat].sort(key=lambda k: indicator_metadata.get(k, {}).get("Full Name", k))
    return {cat: inds for cat, inds in grouped.items() if inds}


def cartopy_available(session_flag_getter=None) -> bool:
    """True if cartopy is installed AND hasn't already failed to reach its
    basemap data this session."""
    if not HAVE_CARTOPY:
        return False
    if session_flag_getter is not None and session_flag_getter():
        return False
    return True


def style_basemap(ax, lon_bounds: tuple[float, float], lat_bounds: tuple[float, float],
                   use_cartopy: bool) -> tuple[bool, Optional[object]]:
    """Apply the shared basemap styling to `ax`. Returns (used_cartopy, transform)."""
    if use_cartopy:
        try:
            with _scoped_socket_timeout(CARTOPY_TIMEOUT_SEC):
                ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.OCEAN, facecolor=BASEMAP_STYLE["ocean_color"], zorder=0)
                ax.add_feature(cfeature.LAND, facecolor=BASEMAP_STYLE["land_color"], zorder=0)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=BASEMAP_STYLE["coastline_color"], zorder=1)
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":",
                                edgecolor=BASEMAP_STYLE["border_color"], zorder=1)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=BASEMAP_STYLE["gridline_color"], alpha=0.6)
                gl.top_labels = gl.right_labels = False
            return True, ccrs.PlateCarree()
        except Exception:
            pass  # fall through to plain-axes styling; caller records the failure

    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_facecolor(BASEMAP_STYLE["ocean_color"])
    ax.grid(alpha=0.3, color=BASEMAP_STYLE["gridline_color"])
    return False, None


def make_figure(use_cartopy: bool, figsize=None, ncols=2, width_ratios=(2.8, 1.0)):
    """Create a 1xN figure, giving GeoAxes only to panels that need them.
    The first panel is always a plain Axes (the chart, never geographic) and
    gets the larger share of the width - it carries the actual signal, the
    locator map only needs to be readable. `figsize` defaults to the active
    text scale's canvas (see TEXT_SCALES) so figure size and font size stay
    matched; pass it explicitly only to override that."""
    if figsize is None:
        figsize = TEXT_SCALES[DEFAULT_TEXT_SCALE]["figsize"]
    ratios = list(width_ratios)[:ncols]
    if len(ratios) < ncols:
        ratios += [1.0] * (ncols - len(ratios))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, ncols, width_ratios=ratios)
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, ncols):
        axes.append(fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
                    if use_cartopy else fig.add_subplot(gs[0, i]))
    return fig, axes


def plot_point_locator(ax, grid_lats, grid_lons, lat_bounds, lon_bounds,
                        sel_lat, sel_lon, in_lat, in_lon, use_cartopy: bool) -> bool:
    """Whole-domain locator map with every grid node shown faintly and the
    selected node highlighted. Returns whether cartopy rendering succeeded."""
    used_cartopy, transform = style_basemap(ax, lon_bounds, lat_bounds, use_cartopy)
    kw = {} if transform is None else {"transform": transform}

    ax.scatter(grid_lons, grid_lats, s=2, color="#888888", alpha=0.35, zorder=2, **kw)
    ax.scatter([sel_lon], [sel_lat], s=90, color="crimson", edgecolor="black", zorder=4,
               label="Grid node", **kw)
    if (sel_lat, sel_lon) != (in_lat, in_lon):
        ax.scatter([in_lon], [in_lat], s=70, marker="x", color="navy", zorder=5,
                   label="Input", **kw)
        ax.plot([in_lon, sel_lon], [in_lat, sel_lat], color="navy", linewidth=0.8,
                linestyle="--", zorder=3, **kw)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=_fs(-2.5),
              markerscale=0.7, handletextpad=0.4, borderpad=0.35)
    ax.set_title("Grid location")
    return used_cartopy if use_cartopy else True


def plot_point_timeseries(ax, years, values, units: str, full_name: str, subtitle: str, color: str = "#c0392b"):
    """Yearly time series with an OLS trend line. If every value is NaN (the
    indicator is genuinely not computable at this location for the selected
    period - e.g. a temperature threshold never reached), draws an explicit
    placeholder with fixed axis limits instead of letting matplotlib
    autoscale from all-NaN data: that autoscale is a known source of
    version-dependent crashes (a NaN/Inf axis span reaching an integer
    conversion deep in the tick locator), not something to just plot and
    hope works."""
    years = np.asarray(years)
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)

    if valid.sum() == 0:
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5) if len(years) else ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "No data for this indicator\nat this location/period",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_fs(-0.5), color=MUTED_TEXT_COLOR)
        ax.set_xlabel("Year")
        ax.set_title(_wrap(f"{full_name} - {subtitle}"))
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.plot(years, values, marker="o", markersize=4, linewidth=1.4, color=color)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(years[valid], values[valid], 1)
        trend = slope * years + intercept
        ax.plot(years, trend, linestyle="--", color="grey", linewidth=1.1,
                label=f"OLS trend: {slope * 10:+.3f} {units}/decade")
        ax.legend(framealpha=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel(_wrap(str(units), 20))
    ax.set_title(_wrap(f"{full_name} - {subtitle}"))
    _style_panel(ax)


# --------------------------------------------------------------------------
# Multi-scenario charts (historical + SSPs on one panel)
# --------------------------------------------------------------------------
# Every function here consumes the long-format frame produced by
# backend.multi_point_series() (columns Scenario, Year, value) or the period
# table from backend.period_means(), so adding a chart type does not touch the
# data layer and adding a scenario workbook does not touch the chart code.

CHART_LINE = "Line - annual time series"
CHART_BARS_ANNUAL = "Bars - annual"
CHART_BARS_PERIOD = "Bars - period means"
CHART_BARS_ANOMALY = "Bars - change vs baseline"
CHART_TYPES = [CHART_LINE, CHART_BARS_ANNUAL, CHART_BARS_PERIOD, CHART_BARS_ANOMALY]

GROUP_WIDTH = 0.82          # fraction of one x-unit occupied by a bar group
MAX_BAR_WIDTH = 0.62        # a lone bar in its slot should not become a block
MARKER_YEAR_LIMIT = 45      # above this many years, markers turn the line to mush
ANNOTATE_BAR_LIMIT = 18     # above this many bars, value labels stop being readable


def _display(display_name: Optional[Callable[[str], str]], label: str) -> str:
    return display_name(label) if display_name else str(label)


def _scenario_order(long_df: pd.DataFrame, column: str = "Scenario") -> list[str]:
    """Scenario labels in the order backend put them (historical first, then
    ascending radiative forcing) - not pandas' alphabetical order, which would
    interleave the historical run with the SSPs."""
    return list(dict.fromkeys(long_df[column].tolist()))


def _no_data_panel(ax, title: str, message: str) -> None:
    """Explicit empty panel with fixed limits.

    Not just a cosmetic nicety: letting matplotlib autoscale an all-NaN series
    yields a non-finite axis span, which fails at draw time deep in the tick
    locator (cannot convert float NaN to integer) rather than where the data
    problem is. Several of these indicators are genuinely not computable at a
    given node (a temperature threshold never reached), so this is a normal
    path, not an error path.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=_fs(-0.5), color=MUTED_TEXT_COLOR)
    ax.set_title(_wrap(title))
    ax.grid(False)


def empty_panel(ax, title: str, message: str) -> None:
    """Public wrapper: render the panel's "nothing to plot" state."""
    _no_data_panel(ax, title, message)


def _integer_year_ticks(ax) -> None:
    """Let matplotlib choose how many year ticks fit, but keep them integers -
    a 53-year window must not print '1992.5'. The tick count adapts to the
    active font size on its own, which is the point of driving sizes from
    rcParams."""
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins="auto"))


def _mark_splice(ax, splice_at: Optional[int]) -> None:
    """Vertical rule where the record changes from the observed-forced
    historical run to scenario-driven projections. The two are not a single
    homogeneous series - the join is a change of experiment, not just of
    period - so it is marked explicitly rather than left implied.

    Labelled in the panel, not in the legend: a legend entry for it costs a
    full row, and at the larger text scales the legend box then grows wider
    than the panel it sits in.
    """
    if splice_at is None:
        return
    ax.axvline(splice_at - 0.5, color="#5a5a5a", linestyle=(0, (4, 3)),
               linewidth=1.1, zorder=1.5)
    ax.annotate(f"{splice_at} ->", xy=(splice_at - 0.5, 0.015), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", rotation=90,
                ha="left", va="bottom", fontsize=_fs(-3), color=MUTED_TEXT_COLOR)


def _hatch_edge_for(color) -> str:
    """Hatch/edge colour with enough contrast against `color`. Matplotlib draws
    a patch's hatch in its edge colour, so a dark hatch on the near-black
    historical bar is invisible - the incomplete-period marking has to flip to
    white there."""
    r, g, b = mpl.colors.to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.45 else "#333333"


def _legend(ax, handles=None) -> None:
    """Compact legend: at most two columns, short labels, inside the panel.

    Explicit `handles` matter for the bar charts: matplotlib takes a bar
    container's legend handle from its first bar, so hatching that bar to mark
    an incomplete period would also hatch the legend swatch and read as part
    of the scenario's identity.
    """
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    n = len(handles)
    if n == 0:
        return
    ax.legend(handles=handles, loc="best", framealpha=0.9,
              ncol=2 if n >= 4 else 1, borderpad=0.4, handlelength=1.6,
              handletextpad=0.5, columnspacing=1.0, labelspacing=0.35)


def _headroom(ax, top_frac: float = 0.16, bottom_frac: float = 0.0) -> None:
    """Pad the y-axis so an inside legend and any bar value labels have
    somewhere to sit instead of overprinting the data."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return
    ax.set_ylim(lo - span * bottom_frac, hi + span * top_frac)


def _slotted_positions(pivot: pd.DataFrame, order: list[str],
                        group_width: float = GROUP_WIDTH) -> dict[str, tuple[list, list, list]]:
    """Grouped-bar geometry, allocating width per x-position rather than
    globally.

    Historical (1985-2014) and the SSPs (2015-) do not overlap in time, so a
    fixed width of group_width/n_scenarios would draw every bar at a quarter
    width with three empty slots beside it. Instead each x-position shares its
    width only among the scenarios that actually have a value there: bars are
    full width where one scenario is present and grouped where several are,
    which is both denser and honest about what is being compared.

    Returns {scenario: (x positions, heights, widths)}, x in pivot-index units.
    """
    out = {sc: ([], [], []) for sc in order}
    for x in pivot.index:
        present = [sc for sc in order if sc in pivot.columns and pd.notna(pivot.at[x, sc])]
        n = len(present)
        if n == 0:
            continue
        width = min(group_width / n, MAX_BAR_WIDTH)
        for i, sc in enumerate(present):
            xs, hs, ws = out[sc]
            xs.append(x + (i - (n - 1) / 2.0) * width)
            hs.append(float(pivot.at[x, sc]))
            ws.append(width)
    return out


def plot_multi_timeseries(ax, long_df: pd.DataFrame, units: str, full_name: str, title: str,
                           display_name: Optional[Callable[[str], str]] = None,
                           show_trend: bool = True,
                           splice_at: Optional[int] = None) -> dict[str, float]:
    """Annual time series for one or more scenarios on a single panel.

    Each scenario keeps its own OLS trend fitted over its own years - fitting
    one line through a spliced historical+projection series would report the
    step between two different experiments as a trend, which it is not.

    Returns {scenario label: OLS slope per decade} for the scenarios actually
    drawn, so the caller can report the numbers as text. They are deliberately
    not written into the legend: four scenario labels each carrying a slope
    make the legend box wider than the panel once the text scale goes up.
    """
    if long_df.empty or long_df["value"].notna().sum() == 0:
        _no_data_panel(ax, title, "No data for this indicator\nat this location / period")
        return {}

    order = _scenario_order(long_df)
    colors = scenario_colors(order)
    n_years = long_df["Year"].nunique()
    marker = "o" if n_years <= MARKER_YEAR_LIMIT else None

    trends: dict[str, float] = {}
    for sc in order:
        g = long_df[long_df["Scenario"] == sc].sort_values("Year")
        years = g["Year"].to_numpy(dtype=float)
        values = g["value"].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() == 0:
            continue
        ax.plot(years, values, marker=marker, markersize=3.6, linewidth=1.6,
                color=colors[sc], label=_display(display_name, sc), zorder=3)
        if valid.sum() >= 3:
            fit = np.polyfit(years[valid], values[valid], 1)
            trends[sc] = float(fit[0]) * 10.0
            if show_trend:
                ax.plot(years[valid], np.polyval(fit, years[valid]), linestyle="--",
                        linewidth=1.1, color=colors[sc], alpha=0.75, zorder=2)

    _mark_splice(ax, splice_at)
    ax.set_xlabel("Year")
    ax.set_ylabel(_wrap(str(units), 20))
    ax.set_title(_wrap(title))
    _integer_year_ticks(ax)
    _style_panel(ax)
    _headroom(ax)
    _legend(ax)
    return trends


def plot_annual_bars(ax, long_df: pd.DataFrame, units: str, full_name: str, title: str,
                      display_name: Optional[Callable[[str], str]] = None,
                      splice_at: Optional[int] = None) -> None:
    """Year-by-year bars, grouped only where scenarios share a year.

    Bars make the year-to-year spread of a count indicator (dry days,
    heatwaves) easier to read off than a line does, and they do not imply
    interpolation between years the way a connected line does - these are
    discrete annual counts, not a continuously sampled signal.
    """
    if long_df.empty or long_df["value"].notna().sum() == 0:
        _no_data_panel(ax, title, "No data for this indicator\nat this location / period")
        return

    order = _scenario_order(long_df)
    colors = scenario_colors(order)
    pivot = (long_df.pivot_table(index="Year", columns="Scenario", values="value", aggfunc="mean")
                    .reindex(columns=order).sort_index())
    geom = _slotted_positions(pivot, order)

    handles = []
    for sc in order:
        xs, hs, ws = geom[sc]
        if not xs:
            continue
        thin = min(ws) < 0.30
        ax.bar(xs, hs, width=ws, color=colors[sc],
               edgecolor="none" if thin else "white", linewidth=0.0 if thin else 0.4, zorder=3)
        handles.append(Patch(facecolor=colors[sc], label=_display(display_name, sc)))

    _mark_splice(ax, splice_at)
    ax.set_xlabel("Year")
    ax.set_ylabel(_wrap(str(units), 20))
    ax.set_title(_wrap(title))
    ax.set_xlim(pivot.index.min() - 0.7, pivot.index.max() + 0.7)
    _integer_year_ticks(ax)
    _style_panel(ax)
    ax.grid(False, axis="x")
    _headroom(ax)
    _legend(ax, handles)


def _period_axis(ax, table: pd.DataFrame, starts: list) -> None:
    """Categorical period axis with the period labels from the table."""
    labels = (table.drop_duplicates("period_start").set_index("period_start")["Period"]
                   .reindex(starts).tolist())
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels(labels, rotation=0 if len(starts) <= 5 else 30,
                       ha="center" if len(starts) <= 5 else "right")
    ax.set_xlabel("Period")


def _period_geometry(table: pd.DataFrame, order: list[str], value_col: str):
    """Grouped-bar geometry over a categorical period axis, plus the lookups
    needed to style each bar (spread, completeness)."""
    starts = sorted(table["period_start"].unique())
    x_index = {start: i for i, start in enumerate(starts)}
    pivot = (table.pivot_table(index="period_start", columns="Scenario",
                                values=value_col, aggfunc="mean")
                  .reindex(index=starts, columns=order))
    pivot.index = [x_index[s] for s in pivot.index]
    geom = _slotted_positions(pivot, order)
    inv_x = {i: s for s, i in x_index.items()}
    return starts, geom, inv_x


def plot_period_bars(ax, table: pd.DataFrame, units: str, full_name: str, title: str,
                      display_name: Optional[Callable[[str], str]] = None,
                      show_spread: bool = True, annotate: bool = True) -> None:
    """Grouped bars of period means (from backend.period_means).

    Error bars are +/- 1 standard deviation of the annual values inside each
    period - interannual variability, NOT a confidence interval on the mean
    and NOT model spread (these workbooks carry a single ensemble-mean field,
    so across-model uncertainty is not available here). Periods only partly
    covered by the selected year range are hatched, because their mean is
    taken over fewer years than the others they sit next to.
    """
    if table.empty:
        _no_data_panel(ax, title, "No data for this indicator\nat this location / period")
        return

    order = _scenario_order(table)
    colors = scenario_colors(order)
    starts, geom, inv_x = _period_geometry(table, order, "mean")
    std_lookup = table.set_index(["Scenario", "period_start"])["std"].to_dict()
    complete_lookup = table.set_index(["Scenario", "period_start"])["complete"].to_dict()
    total_bars = int(sum(len(geom[sc][0]) for sc in order))

    handles = []
    for sc in order:
        xs, hs, ws = geom[sc]
        if not xs:
            continue
        periods = [inv_x[int(round(x))] for x in xs]
        errs = (np.array([std_lookup.get((sc, ps), np.nan) for ps in periods], dtype=float)
                if show_spread else None)
        bars = ax.bar(xs, hs, width=ws, color=colors[sc], edgecolor="white", linewidth=0.5,
                      zorder=3, yerr=errs,
                      error_kw=dict(ecolor="#4a4a4a", elinewidth=0.9, capsize=2.5, zorder=4))
        for bar, ps in zip(bars, periods):
            if not complete_lookup.get((sc, ps), True):
                bar.set_hatch("///")
                bar.set_edgecolor(_hatch_edge_for(colors[sc]))
        if annotate and total_bars <= ANNOTATE_BAR_LIMIT:
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=_fs(-2.5), color=MUTED_TEXT_COLOR)
        handles.append(Patch(facecolor=colors[sc], label=_display(display_name, sc)))

    _period_axis(ax, table, starts)
    ax.set_ylabel(_wrap(str(units), 20))
    ax.set_title(_wrap(title))
    _style_panel(ax)
    ax.grid(False, axis="x")
    _headroom(ax, top_frac=0.20)
    _legend(ax, handles)


def plot_anomaly_bars(ax, table: pd.DataFrame, units: str, full_name: str, title: str,
                       baseline_period: str, mode: str = "absolute",
                       display_name: Optional[Callable[[str], str]] = None,
                       annotate: bool = True) -> None:
    """Grouped bars of period means expressed as a change from the historical
    baseline (from backend.anomaly_vs_baseline).

    With a single scenario on screen the bars are coloured by the sign of the
    change (diverging red/blue), which is the clearest reading when there is no
    scenario to distinguish. With several scenarios the bars keep their AR6
    scenario colours instead, so the same colour means the same scenario in
    every chart in the app; the sign is then read off the zero line.
    """
    if table.empty or table["anomaly"].notna().sum() == 0:
        _no_data_panel(ax, title,
                       "Change vs baseline is not available\nfor this indicator at this location")
        return

    order = _scenario_order(table)
    colors = scenario_colors(order)
    single = len(order) == 1
    starts, geom, inv_x = _period_geometry(table, order, "anomaly")
    complete_lookup = table.set_index(["Scenario", "period_start"])["complete"].to_dict()
    total_bars = int(sum(len(geom[sc][0]) for sc in order))

    handles = []
    for sc in order:
        xs, hs, ws = geom[sc]
        if not xs:
            continue
        periods = [inv_x[int(round(x))] for x in xs]
        if single:
            fills = [ANOMALY_COLORS["positive"] if h > 0 else
                     ANOMALY_COLORS["negative"] if h < 0 else ANOMALY_COLORS["zero"] for h in hs]
        else:
            fills = [colors[sc]] * len(hs)
        bars = ax.bar(xs, hs, width=ws, color=fills, edgecolor="white", linewidth=0.5, zorder=3)
        for bar, ps, fill in zip(bars, periods, fills):
            if not complete_lookup.get((sc, ps), True):
                bar.set_hatch("///")
                bar.set_edgecolor(_hatch_edge_for(fill))
        if annotate and total_bars <= ANNOTATE_BAR_LIMIT:
            fmt = "%+.1f%%" if mode == "percent" else "%+.1f"
            ax.bar_label(bars, fmt=fmt, padding=2, fontsize=_fs(-2.5), color=MUTED_TEXT_COLOR)
        handles.append(Patch(facecolor=colors[sc], label=_display(display_name, sc)))

    ax.axhline(0, color="#1b1b1b", linewidth=1.0, zorder=4)
    _period_axis(ax, table, starts)
    unit_label = "%" if mode == "percent" else units
    ax.set_ylabel(_wrap(f"Change vs {baseline_period}\n[{unit_label}]", 22))
    ax.set_title(_wrap(title))
    _style_panel(ax)
    ax.grid(False, axis="x")
    _headroom(ax, top_frac=0.20, bottom_frac=0.12)
    if not single:
        _legend(ax, handles)
