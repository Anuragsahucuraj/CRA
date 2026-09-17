"""
app.py - Climate Risk Explorer (Grid Point mode), Streamlit frontend
========================================================================
Run with:  streamlit run app.py

UI orchestration only - all data logic lives in backend.py, all rendering
and indicator presentation (categories/colours) lives in viz.py.

Workflow: pick a Scenario, type a Latitude/Longitude, pick an indicator
(grouped by category) and a year range, click Plot. Shows the nearest
actual grid node's District/State, its yearly time series with an OLS
trend, and a locator map.

Data sources: upload the workbooks (one or more) and the mapping CSV via the
sidebar, or place all of them next to this script - any .xlsx file in the
same folder with a valid 'Metadata' sheet is picked up automatically.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import backend as be
import viz

st.set_page_config(page_title="Climate Risk Explorer", page_icon="\U0001F321\uFE0F", layout="wide")

DEFAULT_CSV = Path("points_mapped.csv")
SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")


# --------------------------------------------------------------------------
# Cached wrappers around backend.py (Streamlit-specific; backend.py itself
# has no streamlit dependency)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading indicator workbook (large files can take ~10-20s)...")
def _parse_scenario_workbook_cached(file_bytes: bytes, source_name: str):
    return be.parse_scenario_workbook(file_bytes, source_name)


@st.cache_data(show_spinner="Loading district/state mapping...")
def _load_mapping_cached(file_bytes: bytes) -> pd.DataFrame:
    return be.load_mapping(file_bytes)


@st.cache_data(show_spinner=False)
def _build_grid_lookup_cached(data: pd.DataFrame) -> pd.DataFrame:
    return be.build_grid_lookup(data)


# --------------------------------------------------------------------------
# Sidebar: data sources
# --------------------------------------------------------------------------

st.sidebar.header("\U0001F4C1 Data files")
xlsx_uploads = st.sidebar.file_uploader(
    "Indicator workbooks (.xlsx) - one per scenario", type=["xlsx"], accept_multiple_files=True)
csv_upload = st.sidebar.file_uploader("District/State mapping (.csv)", type=["csv"])

if not viz.HAVE_CARTOPY:
    st.sidebar.warning("cartopy is not installed - the map will show a plain scatter "
                        "with no coastlines/borders. `pip install cartopy` for the full basemap.")
elif st.session_state.get("cartopy_unreachable", False):
    st.sidebar.warning("cartopy could not reach its basemap data this session (no internet, "
                        "or blocked by a firewall/proxy) - showing a plain scatter map instead.")

scenario_sources = {}  # source_name -> file bytes
if xlsx_uploads:
    for f in xlsx_uploads:
        scenario_sources[f.name] = f.getvalue()
else:
    for path in sorted(SCRIPT_DIR.glob("*.xlsx")):
        try:
            file_bytes = path.read_bytes()
            _parse_scenario_workbook_cached(file_bytes, path.name)  # cheap once cached; validates the file
            scenario_sources[path.name] = file_bytes
        except Exception:
            continue  # not one of our workbooks (or unreadable) - skip silently in auto-discovery

csv_bytes = csv_upload.getvalue() if csv_upload is not None else (
    DEFAULT_CSV.read_bytes() if DEFAULT_CSV.exists() else None)

if not scenario_sources or csv_bytes is None:
    st.title("\U0001F321\uFE0F Climate Risk Explorer")
    st.info("Upload at least one indicator workbook (.xlsx) and the district/state "
            "mapping (.csv) in the sidebar to continue - or place them next to "
            f"`app.py` (mapping file named `{DEFAULT_CSV.name}`).")
    st.stop()

scenarios = {}
for name, file_bytes in scenario_sources.items():
    try:
        sc = _parse_scenario_workbook_cached(file_bytes, name)
    except Exception as e:
        import traceback
        st.sidebar.warning(f"Could not read '{name}' as an indicator workbook: {e}")
        with st.sidebar.expander(f"Error detail: {name}"):
            st.code(traceback.format_exc())
        continue
    scenarios[sc.label] = sc

if not scenarios:
    st.error("None of the uploaded/found .xlsx files could be parsed as an indicator workbook.")
    st.stop()

try:
    mapped = _load_mapping_cached(csv_bytes)
except Exception as e:
    st.error(f"Could not read the district/state mapping CSV: {e}")
    st.stop()

admin_lookup = mapped.set_index(["Latitude", "Longitude"])[
    ["State", "District", "State_ID", "District_ID", "match"]
]

# --------------------------------------------------------------------------
# Header + scenario picker
# --------------------------------------------------------------------------

st.title("\U0001F321\uFE0F Climate Risk Explorer")
st.markdown("Grid-point explorer for CMIP6-derived climate indicators across India (0.25\u00b0 resolution).")

scenario_label = st.sidebar.selectbox("\U0001F5D3\uFE0F Scenario", options=sorted(scenarios.keys()))
sc = scenarios[scenario_label]
data, run_metadata, indicator_metadata = sc.data, sc.run_metadata, sc.indicator_metadata
INDICATOR_COLUMNS = sc.indicators
YEAR_MIN, YEAR_MAX = sc.year_min, sc.year_max

info_cols = st.columns(4)
info_cols[0].metric("Scenario", scenario_label.split(" (")[0])
info_cols[1].metric("Period", f"{YEAR_MIN}-{YEAR_MAX}")
info_cols[2].metric("Grid points", f"{data[['Latitude','Longitude']].drop_duplicates().shape[0]:,}")
info_cols[3].metric("Source", run_metadata.get("Source Dataset", "-"))

with st.expander("\u2139\uFE0F About this dataset"):
    st.write(f"**Source dataset:** {run_metadata.get('Source Dataset', '-')}")
    st.write(f"**Spatial resolution:** {run_metadata.get('Spatial Resolution', '-')}")
    st.write(f"**Baseline for percentile indices:** {run_metadata.get('Baseline for Percentile Indices', '-')}")
    st.write(f"**Season restrictions:** {run_metadata.get('Season Restrictions', run_metadata.get('Season Restriction', '-'))}")
    st.write(f"**Generated on:** {run_metadata.get('Generated On', '-')}")
    if len(scenarios) > 1:
        st.write("**All scenarios loaded:** " + ", ".join(sorted(scenarios.keys())))
    st.dataframe(
        pd.DataFrame(indicator_metadata).T[["Full Name", "Units", "Definition", "Calculation Method"]],
        use_container_width=True,
    )

spacing_warning = be.check_grid_spacing(data)
if spacing_warning:
    st.sidebar.warning(f"[{scenario_label}] {spacing_warning}")

n_missing = be.coverage_gap(data, mapped)
if n_missing:
    st.sidebar.warning(f"[{scenario_label}] {n_missing} grid points have no entry in the "
                        f"mapping file - District/State will show as 'Unknown' for these.")

grid_lookup = _build_grid_lookup_cached(data)
LAT_BOUNDS = (data["Latitude"].min() - 0.5, data["Latitude"].max() + 0.5)
LON_BOUNDS = (data["Longitude"].min() - 0.5, data["Longitude"].max() + 0.5)

use_cartopy = viz.cartopy_available(lambda: st.session_state.get("cartopy_unreachable", False))


def _note_cartopy_result(succeeded: bool):
    if use_cartopy and not succeeded:
        st.session_state["cartopy_unreachable"] = True


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

st.divider()
st.subheader("\U0001F4CD Select a grid point")

indicator_groups = viz.group_indicators_by_category(INDICATOR_COLUMNS, indicator_metadata)

defaults_key = f"point_controls::{scenario_label}"
if defaults_key not in st.session_state:
    first_category = next(iter(indicator_groups))
    st.session_state[defaults_key] = dict(
        lat=28.625, lon=77.125, category=first_category, indicator=indicator_groups[first_category][0],
        year_range=(YEAR_MIN, YEAR_MAX),
    )
saved = st.session_state[defaults_key]

with st.form(f"point_form::{scenario_label}"):
    row1 = st.columns(2)
    lat_in = row1[0].number_input("Latitude", value=float(saved["lat"]), step=0.125, format="%.3f")
    lon_in = row1[1].number_input("Longitude", value=float(saved["lon"]), step=0.125, format="%.3f")

    row2 = st.columns(2)
    category = row2[0].selectbox(
        "Indicator category", options=list(indicator_groups.keys()),
        index=list(indicator_groups.keys()).index(saved["category"]) if saved["category"] in indicator_groups else 0,
    )
    category_indicators = indicator_groups[category]
    indicator = row2[1].selectbox(
        "Indicator", options=category_indicators,
        index=category_indicators.index(saved["indicator"]) if saved["indicator"] in category_indicators else 0,
        format_func=lambda k: f"{indicator_metadata[k]['Full Name']} [{indicator_metadata[k]['Units']}]",
    )

    year_range = st.slider("Year range", min_value=YEAR_MIN, max_value=YEAR_MAX, value=saved["year_range"])
    submitted = st.form_submit_button("\U0001F4CA Plot", type="primary", use_container_width=True)

if submitted:
    st.session_state[defaults_key] = dict(
        lat=lat_in, lon=lon_in, category=category, indicator=indicator, year_range=year_range,
    )
ctrl = st.session_state[defaults_key]
lat_in, lon_in, indicator = ctrl["lat"], ctrl["lon"], ctrl["indicator"]
y0, y1 = ctrl["year_range"]

if not (LAT_BOUNDS[0] <= lat_in <= LAT_BOUNDS[1] and LON_BOUNDS[0] <= lon_in <= LON_BOUNDS[1]):
    st.error(f"Input ({lat_in}, {lon_in}) is outside the dataset domain "
             f"(lat {LAT_BOUNDS[0]:.2f} to {LAT_BOUNDS[1]:.2f}, "
             f"lon {LON_BOUNDS[0]:.2f} to {LON_BOUNDS[1]:.2f}). Please re-check.")
    st.stop()

# --------------------------------------------------------------------------
# Resolve grid point and render
# --------------------------------------------------------------------------

m = be.nearest_grid_point(lat_in, lon_in, grid_lookup, admin_lookup)
meta = indicator_metadata[indicator]

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
    st.warning("This grid node has no entry in the mapping file - District/State "
               "attribution is unavailable for it.")

ts = be.point_time_series(data, m.lat, m.lon, indicator, y0, y1)
n_valid = ts[indicator].notna().sum()
if n_valid == 0:
    st.warning(f"**{meta['Full Name']}** is not computable at this location for any year in "
               f"{y0}-{y1} - see its definition below for the condition that must be met "
               f"(e.g. a temperature threshold never reached here). Showing the (empty) chart anyway.")
elif n_valid < len(ts):
    st.caption(f"Note: {len(ts) - n_valid} of {len(ts)} years are blank for this indicator at this "
               f"location - see its definition below for when it's not computable (e.g. a temperature "
               f"threshold never reached).")

fig, (ax_ts, ax_map) = viz.make_figure(use_cartopy)
viz.plot_point_timeseries(
    ax_ts, ts["Year"].values, ts[indicator].values, meta["Units"], meta["Full Name"],
    f"{m.district}, {m.state}  -  {scenario_label}", color=viz.get_colormap_line_color(indicator),
)
succeeded = viz.plot_point_locator(ax_map, grid_lookup["Latitude"].values, grid_lookup["Longitude"].values,
                                    LAT_BOUNDS, LON_BOUNDS, m.lat, m.lon, lat_in, lon_in, use_cartopy)
_note_cartopy_result(succeeded)
fig.tight_layout()
st.pyplot(fig)

st.caption(f"**Definition:** {meta['Definition']}")
st.caption(f"**Method:** {meta['Calculation Method']}")

with st.expander("\U0001F4C4 Data table for this grid node"):
    st.dataframe(ts.rename(columns={indicator: meta["Full Name"]}), use_container_width=True)
