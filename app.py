"""
app.py - Climate Risk Explorer (Grid Point mode), Streamlit frontend
========================================================================
Run with:  streamlit run app.py

UI orchestration only - all data logic lives in backend.py, all rendering
and indicator presentation (categories/colours/typography) lives in viz.py.

Workflow: every control sits in the left sidebar, top to bottom in the order
you use them - upload the indicator workbooks, set the year range, choose
Historical / Projections / Historical + Projections, type a Latitude/Longitude,
pick an indicator (grouped by category), click Plot, then pick a chart type.
The main panel is output only: the nearest actual grid node's District/State,
the requested chart with an OLS trend, a locator map and the data table.

Chart types
-----------
  Line - annual time series     one line per scenario, trend fitted per
                                scenario over its own years
  Bars - annual                 year-by-year bars, grouped only where two
                                scenarios share a year
  Bars - period means           calendar-aligned period means (5/10/20/30 yr)
                                with +/- 1 s.d. interannual variability
  Bars - change vs baseline      period means as a change from the historical
                                reference period, absolute or per cent

Data sources: the district/state mapping (`points_mapped.csv`) ships with the
app and is read from the script's own folder at startup - it is not uploaded.
Indicator workbooks are uploaded (one or more) via the sidebar, or placed next
to this script - any .xlsx file in the same folder with a valid 'Metadata'
sheet is picked up automatically.
"""
import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import backend as be
import viz

st.set_page_config(page_title="Climate Risk App", page_icon="\U0001F321️", layout="wide")

# Resolved against the script's own folder rather than the process working
# directory: `streamlit run` is often launched from elsewhere, and on Streamlit
# Community Cloud the cwd is the repo root, which need not be where app.py sits.
SCRIPT_DIR = (Path(__file__).resolve().parent if "__file__" in dir()
              else Path(".").resolve())
MAPPING_CSV = SCRIPT_DIR / "points_mapped.csv"

PERIOD_LENGTHS = [5, 10, 20, 30]

MODE_HISTORICAL = "Historical"
MODE_PROJECTIONS = "Projections"
MODE_BOTH = "Historical + Projections"

# Opening grid node. Arbitrary but deliberate: a land node inside the domain,
# so the app has something plottable on first load instead of an empty panel.
# (28.625 N, 77.125 E is the Delhi cell.) Change these two numbers to open
# somewhere else.
DEFAULT_LAT, DEFAULT_LON = 17.625, 78.125

# Streamlit's stock sidebar spacing (1rem between blocks, ~6rem of top
# padding, h2-sized section headers) makes five control groups overflow into a
# scroll on a laptop screen. This trims the vertical rhythm only - no colours,
# no widget restyling - and is scoped to the sidebar. It is cosmetic by
# design: if these data-testid hooks are renamed in a future Streamlit the
# rules simply stop matching and the sidebar reverts to the default spacing.
SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    padding-top: 1.1rem;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1.0rem;
    margin: 0.35rem 0 0.05rem 0;
    padding: 0;
}
section[data-testid="stSidebar"] hr { margin: 0.5rem 0; }
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {
    font-size: 0.74rem;
    line-height: 1.25;
    margin-bottom: 0;
}
section[data-testid="stSidebar"] label p { font-size: 0.82rem; margin-bottom: 0.1rem; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary p {
    font-size: 0.86rem;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] {
    padding-top: 1.0rem;
}
</style>
"""


# --------------------------------------------------------------------------
# Cached wrappers around backend.py (Streamlit-specific; backend.py itself
# has no streamlit dependency)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading indicator workbook (large files can take ~10-20s)...")
def _parse_scenario_workbook_cached(file_bytes: bytes, source_name: str):
    return be.parse_scenario_workbook(file_bytes, source_name)


@st.cache_data(show_spinner="Loading district/state mapping...")
def _load_mapping_cached(path_str: str, _mtime_ns: int, _size: int) -> pd.DataFrame:
    """Parse the bundled district/state mapping CSV.

    Keyed on (path, mtime, size) rather than the file's contents: the file
    ships with the app, so re-hashing it on every rerun is pure overhead,
    while the mtime/size pair still invalidates the cache if the bundled
    file is replaced by a re-deployment.
    """
    return be.load_mapping(Path(path_str).read_bytes())


# The leading-underscore arguments below are excluded from Streamlit's cache
# key by design: the workbook's source_name already identifies its contents
# (the parsed frame is itself cached against the file bytes), so hashing a
# ~100k-row DataFrame on every rerun - once per loaded scenario - is pure
# overhead. Keying on the name instead keeps the cache correct and cheap.

@st.cache_data(show_spinner=False)
def _build_grid_lookup_cached(source_name: str, _data: pd.DataFrame) -> pd.DataFrame:
    return be.build_grid_lookup(_data)


@st.cache_data(show_spinner=False)
def _scenario_diagnostics_cached(source_name: str, _data: pd.DataFrame,
                                  _mapped: pd.DataFrame) -> tuple:
    """(grid-spacing warning or None, count of unmapped grid points)."""
    return be.check_grid_spacing(_data), be.coverage_gap(_data, _mapped)


# --------------------------------------------------------------------------
# Figure layout helpers (not cached - these mutate the figure in place)
# --------------------------------------------------------------------------

def _positions_finite(fig) -> bool:
    """True if every Axes in the figure has a finite position rectangle."""
    return all(np.isfinite(ax.get_position().extents).all() for ax in fig.axes)


def _apply_layout(fig, rect=(0.0, 0.0, 1.0, 0.93)) -> bool:
    """tight_layout() that rolls itself back if it yields a non-finite geometry.

    tight_layout() derives the subplot parameters from each Axes' tight bbox.
    A single artist with a non-finite extent - a trend line or annotation
    placed at NaN coordinates when an indicator is not computable at a grid
    node, or a cartopy GeoAxes whose basemap data could not be fetched -
    makes those parameters NaN. Matplotlib does not raise at that point; it
    fails at draw time in MaxNLocator -> Axis.get_tick_space() with
    "ValueError: cannot convert float NaN to integer".

    The default rect leaves the top strip free for the figure-level title
    (location and period), so it never collides with the panel titles.

    Returns True if tight_layout was kept, False if the pre-layout geometry
    was restored.
    """
    saved = [ax.get_position().frozen() for ax in fig.axes]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "Axes not compatible with tight_layout"
            fig.tight_layout(rect=rect)
    except Exception:
        pass
    if _positions_finite(fig):
        return True
    for ax, pos in zip(fig.axes, saved):
        ax.set_position(pos)  # geometry from make_figure(), known finite
    return False


def _figure_png(fig) -> bytes:
    """Publication-resolution PNG of the current figure, for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Sidebar: data sources
# --------------------------------------------------------------------------

st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

# The uploader is the tallest widget in the sidebar and is needed only until
# the workbooks are in, so it is folded away once they are. Auto-discovered
# files are known before the widget is drawn; for uploads, the flag set at the
# bottom of this section reports the previous run's outcome.
_auto_xlsx = sorted(SCRIPT_DIR.glob("*.xlsx"))
_data_ready = bool(_auto_xlsx) or st.session_state.get("workbooks_loaded", False)

with st.sidebar.expander("\U0001F4C1 Data files", expanded=not _data_ready):
    xlsx_uploads = st.file_uploader(
        "Indicator workbooks (.xlsx) - one per scenario",
        type=["xlsx"], accept_multiple_files=True)
    st.caption(f"District/State mapping: `{MAPPING_CSV.name}` (bundled with the app).")

# Advisory notices - basemap availability and per-scenario data-quality
# warnings - collect here, directly under the data section. The container
# decouples the sidebar's visual order from the script's execution order: the
# diagnostics cannot be computed until the scenario selection below is known,
# but they belong with the data, not at the foot of the sidebar.
notices = st.sidebar.container()

if not viz.HAVE_CARTOPY:
    notices.warning("cartopy is not installed - the map will show a plain scatter "
                    "with no coastlines/borders. `pip install cartopy` for the full basemap.")
elif st.session_state.get("cartopy_unreachable", False):
    notices.warning("cartopy could not reach its basemap data this session (no internet, "
                    "or blocked by a firewall/proxy) - showing a plain scatter map instead.")

scenario_sources = {}  # source_name -> file bytes
if xlsx_uploads:
    for f in xlsx_uploads:
        scenario_sources[f.name] = f.getvalue()
else:
    for path in _auto_xlsx:
        try:
            file_bytes = path.read_bytes()
            _parse_scenario_workbook_cached(file_bytes, path.name)  # cheap once cached; validates the file
            scenario_sources[path.name] = file_bytes
        except Exception:
            continue  # not one of our workbooks (or unreadable) - skip silently in auto-discovery

# The mapping is part of the deployment, so a missing or unreadable file is a
# packaging fault, not something the user can fix in the UI - say so plainly and
# stop here rather than surfacing a traceback further down.
if not MAPPING_CSV.exists():
    st.title("\U0001F321️ Climate Risk Explorer")
    st.error(f"The district/state mapping file `{MAPPING_CSV.name}` is missing from this "
             "deployment. It is expected to sit next to `app.py`; re-deploy with the "
             "file included. District/State attribution cannot be done without it.")
    st.stop()

try:
    _mapping_stat = MAPPING_CSV.stat()
    mapped = _load_mapping_cached(str(MAPPING_CSV), _mapping_stat.st_mtime_ns,
                                  _mapping_stat.st_size)
except Exception as e:
    st.title("\U0001F321️ Climate Risk Explorer")
    st.error(f"The bundled district/state mapping file `{MAPPING_CSV.name}` could not be "
             f"read: {e}. Expected columns: Latitude, Longitude, DIST_LGD, DISTRICT, "
             "STATE_LGD, STATE_UT, match.")
    st.stop()

if not scenario_sources:
    st.title("\U0001F321️ Climate Risk Explorer")
    st.info("Upload at least one indicator workbook (.xlsx) in the sidebar to continue - "
            "or place the workbooks next to `app.py`.")
    st.stop()

scenarios = {}
for name, file_bytes in scenario_sources.items():
    try:
        sc = _parse_scenario_workbook_cached(file_bytes, name)
    except Exception as e:
        import traceback
        notices.warning(f"Could not read '{name}' as an indicator workbook: {e}")
        with notices.expander(f"Error detail: {name}"):
            st.code(traceback.format_exc())
        continue
    scenarios[sc.label] = sc

if not scenarios:
    st.error("None of the uploaded/found .xlsx files could be parsed as an indicator workbook.")
    st.stop()

st.session_state["workbooks_loaded"] = True

admin_lookup = mapped.set_index(["Latitude", "Longitude"])[
    ["State", "District", "State_ID", "District_ID", "match"]
]

all_labels = be.sort_scenario_labels(scenarios.keys())
historical_labels = be.sort_scenario_labels([l for l in all_labels if be.is_historical(l)])
projection_labels = be.sort_scenario_labels([l for l in all_labels if not be.is_historical(l)])

# --------------------------------------------------------------------------
# Sidebar: what to show - Historical, Projections, or both
# --------------------------------------------------------------------------
# One three-way choice replaces the earlier free-form multiselect, which
# defaulted to every loaded scenario at once and so always opened with all
# four chips pre-selected. Modes are offered only where the loaded workbooks
# can satisfy them, so the control can never ask for a run that is absent.

modes = []
if historical_labels:
    modes.append(MODE_HISTORICAL)
if projection_labels:
    modes.append(MODE_PROJECTIONS)
if historical_labels and projection_labels:
    modes.append(MODE_BOTH)

st.sidebar.subheader("\U0001F5D3️ Select")
scenario_mode = st.sidebar.radio(
    "Scenarios to show", options=modes,
    index=modes.index(MODE_HISTORICAL) if MODE_HISTORICAL in modes else 0,
    label_visibility="collapsed",
    help="Historical - the observed-forced run on its own.  Projections - the SSP runs "
         "only.  Historical + Projections - both on one panel, historical in black and "
         "each SSP in its IPCC AR6 colour.",
)

if scenario_mode == MODE_HISTORICAL:
    selected_labels = list(historical_labels)
else:
    chosen_projections = list(projection_labels)
    if len(projection_labels) > 1:
        # Kept so a single SSP can still be examined on its own; all of them
        # are shown unless the user narrows the set here.
        with st.sidebar.expander("Which projections?"):
            chosen_projections = st.multiselect(
                "SSP scenarios", options=projection_labels, default=projection_labels,
                format_func=be.scenario_display_name, label_visibility="collapsed",
                help="All loaded SSPs are shown by default. Deselect to compare fewer.",
            )
    selected_labels = ((list(historical_labels) if scenario_mode == MODE_BOTH else [])
                       + list(chosen_projections))

if not selected_labels:
    st.title("\U0001F321️ Climate Risk Explorer")
    st.info("No scenario is selected - pick at least one SSP under 'Which projections?' "
            "in the sidebar, or switch to Historical.")
    st.stop()

selected_labels = be.sort_scenario_labels(selected_labels)
selected = {label: scenarios[label] for label in selected_labels}
# The historical run is the natural reference for metadata and for the grid,
# so it leads when it is on screen; otherwise the lowest-forcing SSP does.
primary = selected[selected_labels[0]]

has_historical = any(be.is_historical(label) for label in selected_labels)
historical_loaded = bool(historical_labels)

# --------------------------------------------------------------------------
# Sidebar: year range
# --------------------------------------------------------------------------
# Placed directly under the scenario choice so the slider always spans exactly
# the period the selected run(s) cover: Historical shows the historical years,
# Projections the projection years, and Historical + Projections the full
# record. The endpoints rescale when the mode changes, and any stored window is
# clamped back into the new bounds rather than left dangling.

SPAN_MIN = min(sc.year_min for sc in selected.values())
SPAN_MAX = max(sc.year_max for sc in selected.values())

st.sidebar.subheader("\U0001F4C5 Year range")

YEAR_KEY = "year_range"
if SPAN_MAX > SPAN_MIN:
    _stored = st.session_state.get(YEAR_KEY, (SPAN_MIN, SPAN_MAX))
    _lo = max(SPAN_MIN, min(int(_stored[0]), SPAN_MAX))
    _hi = min(SPAN_MAX, max(int(_stored[1]), SPAN_MIN))
    if _hi < _lo:
        _lo, _hi = SPAN_MIN, SPAN_MAX
    year_range = st.sidebar.slider(
        "Years", min_value=SPAN_MIN, max_value=SPAN_MAX, value=(_lo, _hi),
        label_visibility="collapsed",
        help="Spans the period the selected scenarios cover; it rescales when you "
             "change the Historical / Projections choice above.",
    )
    st.session_state[YEAR_KEY] = year_range
else:
    # A single-year selection: a range slider cannot be built, and there is
    # nothing to choose.
    year_range = (SPAN_MIN, SPAN_MAX)
    st.sidebar.caption(f"Single year in the data: {SPAN_MIN}.")

# An indicator is only offered if every selected scenario carries it - a
# half-drawn comparison (three scenarios plotted, one silently absent) is
# worse than not offering the indicator. Order and metadata come from the
# primary scenario so the picker does not reshuffle as scenarios are toggled.
common_indicators = set.intersection(*[set(sc.indicators) for sc in selected.values()])
INDICATOR_COLUMNS = [k for k in primary.indicators if k in common_indicators]
indicator_metadata = primary.indicator_metadata

if not INDICATOR_COLUMNS:
    st.error("The selected scenarios share no common indicator - narrow the projection "
             "set, or plot the scenarios one at a time.")
    st.stop()

dropped = [k for sc in selected.values() for k in sc.indicators if k not in common_indicators]
if dropped:
    st.sidebar.caption(f"{len(set(dropped))} indicator(s) missing from one or more selected "
                       "scenarios are hidden from the picker.")

YEAR_MIN = min(sc.year_min for sc in selected.values())
YEAR_MAX = max(sc.year_max for sc in selected.values())

# The requested window, clipped to what the selected scenarios cover, so that
# the figure title and the CSV filename state the period actually plotted
# rather than the slider's endpoints.
y0, y1 = max(year_range[0], YEAR_MIN), min(year_range[1], YEAR_MAX)
if y1 < y0:
    st.title("\U0001F321️ Climate Risk Explorer")
    st.info(f"The selected year range ({year_range[0]}-{year_range[1]}) does not overlap "
            f"the period covered by **{scenario_mode}** ({YEAR_MIN}-{YEAR_MAX}). Widen the "
            "year range or change the scenario selection.")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar: grid point and indicator
# --------------------------------------------------------------------------
# In the sidebar rather than the main panel so that every input lives in one
# column and the main panel is output only. It must follow the scenario
# selection above, because the indicator list is derived from the selected
# scenarios.

indicator_groups = viz.group_indicators_by_category(INDICATOR_COLUMNS, indicator_metadata)

# One control state across scenario selections, clamped to whatever is
# currently valid - switching mode changes the indicator set, and silently
# keeping a value that is no longer offered would crash the widget.
DEFAULTS_KEY = "point_controls"
if DEFAULTS_KEY not in st.session_state:
    first_category = next(iter(indicator_groups))
    st.session_state[DEFAULTS_KEY] = dict(
        lat=DEFAULT_LAT, lon=DEFAULT_LON, category=first_category,
        indicator=indicator_groups[first_category][0],
    )
saved = dict(st.session_state[DEFAULTS_KEY])
if saved["category"] not in indicator_groups:
    saved["category"] = next(iter(indicator_groups))
if saved["indicator"] not in indicator_groups[saved["category"]]:
    saved["indicator"] = indicator_groups[saved["category"]][0]

st.sidebar.subheader("\U0001F4CD Grid point")

with st.sidebar.form("point_form"):
    # Two columns: the pair of coordinates is the one place in the sidebar
    # where side-by-side fields still read cleanly, and it saves a row.
    coords = st.columns(2)
    lat_in = coords[0].number_input("Lat", value=float(saved["lat"]), step=0.125, format="%.3f")
    lon_in = coords[1].number_input("Lon", value=float(saved["lon"]), step=0.125, format="%.3f")

    category = st.selectbox(
        "Indicator category", options=list(indicator_groups.keys()),
        index=list(indicator_groups.keys()).index(saved["category"]),
    )
    category_indicators = indicator_groups[category]
    indicator = st.selectbox(
        "Indicator", options=category_indicators,
        index=category_indicators.index(saved["indicator"]) if saved["indicator"] in category_indicators else 0,
        format_func=lambda k: f"{indicator_metadata[k]['Full Name']} [{indicator_metadata[k]['Units']}]",
    )
    submitted = st.form_submit_button("\U0001F4CA Plot", type="primary", width="stretch")

if submitted:
    saved = dict(lat=lat_in, lon=lon_in, category=category, indicator=indicator)
st.session_state[DEFAULTS_KEY] = saved

lat_in, lon_in, indicator = saved["lat"], saved["lon"], saved["indicator"]

# --------------------------------------------------------------------------
# Sidebar: chart type and appearance
# --------------------------------------------------------------------------
# These live outside the form on purpose. Widgets inside a Streamlit form do
# not rerun the script until the form is submitted, so a chart-type switch
# placed there would not be able to reveal its own dependent options (period
# length, anomaly mode) until after a second submit. Out here, changing the
# chart type re-renders immediately from the already-chosen grid point.

st.sidebar.subheader("\U0001F4C8 Chart")
chart_type = st.sidebar.radio("Chart type", options=viz.CHART_TYPES, index=0,
                              label_visibility="collapsed")

show_trend = True
period_len = 10
show_spread = True
anomaly_mode = "absolute"

if chart_type == viz.CHART_LINE:
    show_trend = st.sidebar.checkbox("Show OLS trend per scenario", value=True)
elif chart_type in (viz.CHART_BARS_PERIOD, viz.CHART_BARS_ANOMALY):
    period_len = st.sidebar.selectbox(
        "Averaging period (years)", options=PERIOD_LENGTHS, index=PERIOD_LENGTHS.index(10),
        help="Periods are anchored to calendar boundaries (1990-1999, 2000-2009, ...) "
             "so the same bar stays comparable between sessions and scenarios.",
    )
    if chart_type == viz.CHART_BARS_PERIOD:
        show_spread = st.sidebar.checkbox(
            "Show interannual spread (+/- 1 s.d.)", value=True,
            help="Standard deviation of the annual values within each period. "
                 "This is interannual variability, not model uncertainty.",
        )
    else:
        anomaly_mode = st.sidebar.radio(
            "Express change as", options=["absolute", "percent"], index=0,
            format_func=lambda m: "Absolute (indicator units)" if m == "absolute" else "Per cent of baseline",
        )

with st.sidebar.expander("\U0001F58B️ Appearance"):
    text_scale = st.select_slider(
        "Chart text size", options=list(viz.TEXT_SCALES.keys()), value=viz.DEFAULT_TEXT_SCALE,
        help="Sets every text size in the figure from one base size (ticks, axis labels, "
             "legend, titles) and matches the canvas size to it, so labels stay legible "
             "instead of being shrunk by the browser's rescaling.",
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("\U0001F321️ Climate Risk Explorer")
st.markdown("Grid-point explorer for CMIP6-derived climate indicators across India (0.25° resolution).")

info_cols = st.columns(4)
info_cols[0].metric("Scenarios shown", f"{len(selected_labels)} of {len(scenarios)}")
info_cols[1].metric("Combined period", f"{YEAR_MIN}-{YEAR_MAX}")
info_cols[2].metric(
    "Grid points",
    f"{primary.data[['Latitude', 'Longitude']].drop_duplicates().shape[0]:,}")
info_cols[3].metric("Source", primary.run_metadata.get("Source Dataset", "-"))
st.caption(f"Showing **{scenario_mode}**: "
           + ", ".join(be.scenario_display_name(l) for l in selected_labels))
if (y0, y1) != (year_range[0], year_range[1]):
    st.caption(f"Year range clipped from {year_range[0]}-{year_range[1]} to {y0}-{y1}, "
               "the period the selected scenarios cover.")

with st.expander("ℹ️ About this dataset"):
    st.write(f"**Source dataset:** {primary.run_metadata.get('Source Dataset', '-')}")
    st.write(f"**Spatial resolution:** {primary.run_metadata.get('Spatial Resolution', '-')}")
    st.write(f"**Baseline for percentile indices:** "
             f"{primary.run_metadata.get('Baseline for Percentile Indices', '-')}")
    st.write(f"**Season restrictions:** "
             f"{primary.run_metadata.get('Season Restrictions', primary.run_metadata.get('Season Restriction', '-'))}")
    st.write(f"**Generated on:** {primary.run_metadata.get('Generated On', '-')}")
    st.write("**Scenarios loaded:** " + ", ".join(
        f"{be.scenario_display_name(l)} [{scenarios[l].year_min}-{scenarios[l].year_max}]"
        for l in all_labels))
    st.dataframe(
        pd.DataFrame(indicator_metadata).T[["Full Name", "Units", "Definition", "Calculation Method"]],
        width="stretch",
    )

for label in selected_labels:
    spacing_warning, n_missing = _scenario_diagnostics_cached(
        scenarios[label].source_name, scenarios[label].data, mapped)
    if spacing_warning:
        notices.warning(f"[{be.scenario_display_name(label)}] {spacing_warning}")
    if n_missing:
        notices.warning(f"[{be.scenario_display_name(label)}] {n_missing} grid points have no "
                        f"entry in `{MAPPING_CSV.name}` - District/State will show as "
                        "'Unknown' for these.")

grid_lookup = _build_grid_lookup_cached(primary.source_name, primary.data)
LAT_BOUNDS = (min(sc.data["Latitude"].min() for sc in selected.values()) - 0.5,
              max(sc.data["Latitude"].max() for sc in selected.values()) + 0.5)
LON_BOUNDS = (min(sc.data["Longitude"].min() for sc in selected.values()) - 0.5,
              max(sc.data["Longitude"].max() for sc in selected.values()) + 0.5)

use_cartopy = viz.cartopy_available(lambda: st.session_state.get("cartopy_unreachable", False))


def _note_cartopy_result(succeeded: bool):
    if use_cartopy and not succeeded:
        st.session_state["cartopy_unreachable"] = True


if not (LAT_BOUNDS[0] <= lat_in <= LAT_BOUNDS[1] and LON_BOUNDS[0] <= lon_in <= LON_BOUNDS[1]):
    st.error(f"Input ({lat_in}, {lon_in}) is outside the dataset domain "
             f"(lat {LAT_BOUNDS[0]:.2f} to {LAT_BOUNDS[1]:.2f}, "
             f"lon {LON_BOUNDS[0]:.2f} to {LON_BOUNDS[1]:.2f}). Please re-check.")
    st.stop()

# --------------------------------------------------------------------------
# Resolve grid point
# --------------------------------------------------------------------------

m = be.nearest_grid_point(lat_in, lon_in, grid_lookup, admin_lookup)
meta = indicator_metadata[indicator]
units = meta["Units"]

st.divider()
st.subheader(f"\U0001F4CD {m.district}, {m.state}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nearest grid node", f"{m.lat:.3f}, {m.lon:.3f}")
c2.metric("Snap distance", f"{m.distance_km:.1f} km")
c3.metric("District", m.district)
c4.metric("State", m.state)
c5.metric("District / State ID", f"{m.district_id} / {m.state_id}")

if m.distance_km > be.SNAP_WARN_KM:
    st.warning(f"Nearest grid node is {m.distance_km:.1f} km away, more than one grid-cell "
               f"diagonal (~{be.SNAP_WARN_KM:.1f} km) - your input may fall outside the "
               f"land-grid mask (e.g. open sea) or well outside India.")
if m.admin_match == "overlap":
    st.info("This grid node's District/State was assigned by a nearest-district fallback "
            "(coastal/border cell) rather than an exact point-in-polygon match.")
elif m.admin_match == "Unknown":
    st.warning(f"This grid node has no entry in `{MAPPING_CSV.name}` - District/State "
               "attribution is unavailable for it.")

# --------------------------------------------------------------------------
# Extract the series for every selected scenario
# --------------------------------------------------------------------------

long_df = be.multi_point_series(selected, m.lat, m.lon, indicator, y0, y1)
split_year = be.splice_year(long_df)

per_scenario_valid = (long_df.assign(ok=long_df["value"].notna())
                             .groupby("Scenario")["ok"].agg(["sum", "count"])
                      if not long_df.empty else pd.DataFrame(columns=["sum", "count"]))
empty_scenarios = [l for l in selected_labels
                   if l not in per_scenario_valid.index or per_scenario_valid.at[l, "sum"] == 0]
partial_scenarios = [l for l in selected_labels
                     if l in per_scenario_valid.index
                     and 0 < per_scenario_valid.at[l, "sum"] < per_scenario_valid.at[l, "count"]]

if len(empty_scenarios) == len(selected_labels):
    st.warning(f"**{meta['Full Name']}** is not computable at this location for any year in "
               f"{y0}-{y1}, in any selected scenario - see its definition below for the "
               "condition that must be met (e.g. a temperature threshold never reached). "
               "Showing the (empty) chart anyway.")
elif empty_scenarios:
    st.info("No valid years for " + ", ".join(be.scenario_display_name(l) for l in empty_scenarios)
            + " at this location - those scenarios are absent from the chart.")
if partial_scenarios:
    st.caption("Note: some years are blank for "
               + ", ".join(be.scenario_display_name(l) for l in partial_scenarios)
               + " at this location - see the indicator definition below for when it is "
                 "not computable.")

# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------

type_cfg = viz.apply_typography(text_scale)   # must precede make_figure()
fig, (ax_chart, ax_map) = viz.make_figure(use_cartopy, figsize=type_cfg["figsize"])

period_note = None
trend_note = None
baseline = None


def _incomplete_note(period_table: pd.DataFrame) -> "str | None":
    """Name the periods whose mean rests on fewer years than the rest, since
    those bars are hatched on the chart but the reason is not self-evident."""
    if period_table.empty or period_table["complete"].all():
        return None
    incomplete = period_table.loc[~period_table["complete"], "Period"].unique()
    return ("Hatched bars are periods only partly covered by the data or the selected year "
            f"range ({', '.join(incomplete)}), so their mean is taken over fewer years "
            "than the others.")


if chart_type == viz.CHART_LINE:
    trends = viz.plot_multi_timeseries(
        ax_chart, long_df, units=units, full_name=meta["Full Name"],
        title=meta["Full Name"], display_name=be.scenario_short_name,
        show_trend=show_trend, splice_at=split_year,
    )
    if show_trend and trends:
        trend_note = "OLS trend per decade: " + ";  ".join(
            f"{be.scenario_short_name(l)} {v:+.2f} {units}" for l, v in trends.items())
elif chart_type == viz.CHART_BARS_ANNUAL:
    viz.plot_annual_bars(
        ax_chart, long_df, units=units, full_name=meta["Full Name"],
        title=meta["Full Name"], display_name=be.scenario_short_name,
        splice_at=split_year,
    )
elif chart_type == viz.CHART_BARS_PERIOD:
    period_table = be.period_means(long_df, period_len=period_len)
    viz.plot_period_bars(
        ax_chart, period_table, units=units, full_name=meta["Full Name"],
        title=f"{meta['Full Name']} - {period_len}-year means",
        display_name=be.scenario_short_name, show_spread=show_spread,
    )
    period_note = _incomplete_note(period_table)
    if show_spread:
        period_note = ((period_note + " ") if period_note else "") + (
            "Error bars are +/- 1 standard deviation of the annual values within each period "
            "(interannual variability at this grid node) - not model spread, which a single "
            "ensemble-mean field cannot provide.")
else:  # viz.CHART_BARS_ANOMALY
    baseline = be.historical_baseline(scenarios, m.lat, m.lon, indicator)
    if baseline is None:
        viz.empty_panel(
            ax_chart, meta["Full Name"],
            "No historical run loaded, or the indicator\nhas no valid historical year here -\n"
            "change vs baseline is undefined")
        period_note = ("A change vs baseline needs a historical workbook that has valid values "
                       "for this indicator at this grid node. "
                       + ("Load Historical_Master_Indicators.xlsx to enable it."
                          if not historical_loaded else
                          "This indicator has no valid historical year at this node."))
    else:
        period_table = be.period_means(long_df, period_len=period_len)
        anomaly_table = be.anomaly_vs_baseline(period_table, baseline, mode=anomaly_mode)
        viz.plot_anomaly_bars(
            ax_chart, anomaly_table, units=units, full_name=meta["Full Name"],
            title=f"{meta['Full Name']} - change vs {baseline.period}",
            baseline_period=baseline.period, mode=anomaly_mode,
            display_name=be.scenario_short_name,
        )
        incomplete_note = _incomplete_note(anomaly_table)
        period_note = (f"Baseline: {be.scenario_display_name(baseline.label)} mean over "
                       f"{baseline.period} at this grid node = {baseline.mean:.3f} {units} "
                       f"({baseline.n_years} years). The reference period is fixed to the "
                       "historical run's own span, independent of the year range selected above.")
        if anomaly_mode == "percent" and anomaly_table["anomaly"].isna().all():
            period_note += (" Per-cent change is not shown because the baseline is effectively "
                            "zero here - use the absolute change instead.")
        if incomplete_note:
            period_note += " " + incomplete_note

succeeded = viz.plot_point_locator(ax_map, grid_lookup["Latitude"].values, grid_lookup["Longitude"].values,
                                    LAT_BOUNDS, LON_BOUNDS, m.lat, m.lon, lat_in, lon_in, use_cartopy)
_note_cartopy_result(succeeded)

viz.add_figure_title(fig, f"{m.district}, {m.state}  |  {m.lat:.3f}, {m.lon:.3f}  |  {y0}-{y1}")
if not _apply_layout(fig):
    st.caption("Note: automatic figure layout was skipped for this selection "
               "(non-finite element in the plot); using the default geometry.")
st.pyplot(fig)
png_bytes = _figure_png(fig)
plt.close(fig)  # Streamlit re-runs the whole script per interaction

if trend_note:
    st.caption(trend_note)
if period_note:
    st.caption(period_note)
if chart_type in (viz.CHART_LINE, viz.CHART_BARS_ANNUAL) and split_year is not None:
    st.caption(f"The dashed rule at {split_year} marks where the record changes from the "
               "observed-forced historical run to scenario-driven projections; they are "
               "different experiments, not one continuous series."
               + (" Each scenario's OLS trend is fitted over its own years only, for the "
                  "same reason." if chart_type == viz.CHART_LINE and show_trend else ""))
st.caption(f"**Definition:** {meta['Definition']}")
st.caption(f"**Method:** {meta['Calculation Method']}")

# --------------------------------------------------------------------------
# Data table and downloads
# --------------------------------------------------------------------------

present_labels = [l for l in selected_labels if l in set(long_df["Scenario"])] if not long_df.empty else []
wide = (long_df.pivot_table(index="Year", columns="Scenario", values="value", aggfunc="mean")
               .reindex(columns=present_labels)
               .rename(columns=be.scenario_display_name)
        if present_labels else pd.DataFrame())

with st.expander("\U0001F4C4 Data table for this grid node"):
    if wide.empty:
        st.write("No values to show for this indicator at this grid node.")
    else:
        st.caption(f"{meta['Full Name']} [{units}] at {m.lat:.3f}, {m.lon:.3f}")
        st.dataframe(wide, width="stretch")

dl = st.columns(2)
if not wide.empty:
    csv_name = (f"{indicator}_{m.lat:.3f}_{m.lon:.3f}_{y0}-{y1}.csv").replace(" ", "_")
    dl[0].download_button("⬇️ Download plotted data (CSV)",
                          data=wide.to_csv().encode("utf-8"),
                          file_name=csv_name, mime="text/csv", width="stretch")
dl[1].download_button("⬇️ Download figure (PNG, 300 dpi)", data=png_bytes,
                      file_name=f"{indicator}_{m.lat:.3f}_{m.lon:.3f}.png",
                      mime="image/png", width="stretch")
