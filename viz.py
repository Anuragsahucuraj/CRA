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
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

# cartopy's feature downloader has no timeout of its own - on a firewalled or
# slow connection, fetching Natural Earth data on first use can hang
# indefinitely. Cap it so a network problem fails fast instead of freezing
# the app; callers should also remember a failure for the rest of the
# session (see app.py) rather than retrying every render.
CARTOPY_TIMEOUT_SEC = 8
socket.setdefaulttimeout(CARTOPY_TIMEOUT_SEC)

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


def make_figure(use_cartopy: bool, figsize=(13, 5), ncols=2):
    """Create a 1xN figure, giving GeoAxes only to panels that need them.
    The first panel is always a plain Axes (time series, never geographic)."""
    if use_cartopy:
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(1, ncols, 1)]
        for i in range(2, ncols + 1):
            axes.append(fig.add_subplot(1, ncols, i, projection=ccrs.PlateCarree()))
        return fig, axes
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    return fig, list(np.atleast_1d(axes))


def plot_point_locator(ax, grid_lats, grid_lons, lat_bounds, lon_bounds,
                        sel_lat, sel_lon, in_lat, in_lon, use_cartopy: bool) -> bool:
    """Whole-domain locator map with every grid node shown faintly and the
    selected node highlighted. Returns whether cartopy rendering succeeded."""
    used_cartopy, transform = style_basemap(ax, lon_bounds, lat_bounds, use_cartopy)
    kw = {} if transform is None else {"transform": transform}

    ax.scatter(grid_lons, grid_lats, s=2, color="#888888", alpha=0.35, zorder=2, **kw)
    ax.scatter([sel_lon], [sel_lat], s=90, color="crimson", edgecolor="black", zorder=4,
               label="Nearest grid node", **kw)
    if (sel_lat, sel_lon) != (in_lat, in_lon):
        ax.scatter([in_lon], [in_lat], s=70, marker="x", color="navy", zorder=5,
                   label="Your input", **kw)
        ax.plot([in_lon, sel_lon], [in_lat, sel_lat], color="navy", linewidth=0.8,
                linestyle="--", zorder=3, **kw)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Grid location", fontsize=11)
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
                transform=ax.transAxes, ha="center", va="center", fontsize=10, color="grey")
        ax.set_xlabel("Year")
        ax.set_title(f"{full_name}\n{subtitle}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.plot(years, values, marker="o", markersize=4, linewidth=1.4, color=color)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(years[valid], values[valid], 1)
        trend = slope * years + intercept
        ax.plot(years, trend, linestyle="--", color="grey", linewidth=1.1,
                label=f"OLS trend: {slope * 10:+.3f} {units}/decade")
        ax.legend(fontsize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{full_name} [{units}]")
    ax.set_title(f"{full_name}\n{subtitle}", fontsize=10)
    ax.grid(alpha=0.3)
