"""
backend.py - Data layer for the Climate Risk (Heatwave / Heat Index) Explorer
==============================================================================
All I/O and data-wrangling lives here, independent of Streamlit or plotting,
so it can be unit-tested and reused (e.g. in a notebook) without pulling in
the UI. Nothing in this module imports streamlit or matplotlib.

Data model
----------
Each scenario workbook (Historical, SSP2-4.5, SSP3-7.0, SSP5-8.5, ...) has:
  - one sheet per State/UT, long format: Latitude, Longitude, Year, <indicators...>
  - a 'Metadata' sheet: a run-metadata block (Field/Value pairs) followed by a
    per-indicator table (Variable Name/Full Name/Definition/Units/Calculation Method)

points_mapped.csv carries the authoritative, scenario-independent District/State
attribution per grid node (from an exact point-in-polygon join against the
official district/state boundary shapefiles; 'match' = 'point' for an exact
join or 'overlap' for a nearest-district fallback on coastal/border cells
where no polygon reached the grid centroid).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import openpyxl

EARTH_RADIUS_KM = 6371.0088          # IUGG mean radius, for haversine distance
EXPECTED_GRID_SPACING_DEG = 0.25     # nominal resolution, sanity-checked at load time
SNAP_WARN_KM = EXPECTED_GRID_SPACING_DEG * np.sqrt(2) * 111.0  # ~ one grid-cell diagonal


def _cell(row, i):
    """Safely index into a spreadsheet row that may be a short/empty tuple or
    list. openpyxl's read-only mode can return a genuinely empty `[]` (not a
    None-padded tuple) for some blank rows; plain row[i] indexing crashes on
    those. Every raw-row access in this module goes through this helper."""
    return row[i] if i < len(row) else None


@dataclass
class Scenario:
    """One loaded indicator workbook."""
    label: str                      # e.g. "Projection - SSP370 (2015-2037)"
    source_name: str                # original filename, for diagnostics
    run_metadata: dict
    indicator_metadata: dict        # {var_name: {Full Name, Definition, Units, Calculation Method}}
    data: pd.DataFrame              # Sheet_State, Latitude, Longitude, Year, <indicators...>
    year_min: int
    year_max: int

    @property
    def indicators(self) -> list[str]:
        return list(self.indicator_metadata.keys())


def parse_scenario_workbook(file_bytes: bytes, source_name: str) -> Scenario:
    """Parse one indicator workbook. Raises ValueError with a clear message if
    the file isn't shaped like one of these workbooks (missing 'Metadata'
    sheet, no 'Variable Name' header row, etc.) - callers should catch this
    and skip the file rather than crash the whole app."""
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception as e:
        raise ValueError(f"Not a readable .xlsx file: {e}") from e

    if "Metadata" not in wb.sheetnames:
        raise ValueError("No 'Metadata' sheet found.")
    ws = wb["Metadata"]
    rows = list(ws.iter_rows(values_only=True))
    try:
        header_row_idx = next(i for i, r in enumerate(rows) if _cell(r, 0) == "Variable Name")
    except StopIteration:
        raise ValueError("'Metadata' sheet has no 'Variable Name' header row.")
    header = rows[header_row_idx]

    indicator_metadata = {}
    for r in rows[header_row_idx + 1:]:
        name = _cell(r, 0)
        if name is None:
            continue
        indicator_metadata[name] = {
            _cell(header, j + 1): _cell(r, j + 1) for j in range(len(header) - 1)
        }
    if not indicator_metadata:
        raise ValueError("No indicator rows found under the 'Variable Name' header.")

    run_metadata = {
        _cell(r, 0): _cell(r, 1) for r in rows[:header_row_idx] if _cell(r, 0) is not None
    }

    frames = []
    for sheet_name in wb.sheetnames:
        if sheet_name == "Metadata":
            continue
        sheet_ws = wb[sheet_name]
        rows_iter = sheet_ws.iter_rows(values_only=True)
        try:
            sheet_header = next(rows_iter)
        except StopIteration:
            continue  # genuinely empty sheet
        # Some sheets carry trailing blank header cells (leftover formatting/
        # merged-cell artifacts, not real columns) - trim them so the header
        # doesn't end up with duplicate None column names, which breaks
        # pandas' column alignment when concatenating across sheets.
        while sheet_header and sheet_header[-1] is None:
            sheet_header = sheet_header[:-1]
        n_cols = len(sheet_header)
        data_rows = []
        for r in rows_iter:
            if _cell(r, 0) is None:
                continue
            # openpyxl can return a shorter tuple for a row than the header's
            # width when that row's trailing cells were never written (e.g. a
            # blank/NaN indicator value at the end) - pad with None so every
            # row matches the header length before building the DataFrame.
            if len(r) < n_cols:
                r = tuple(r) + (None,) * (n_cols - len(r))
            elif len(r) > n_cols:
                r = r[:n_cols]
            data_rows.append(r)
        if not data_rows:
            continue
        sheet_df = pd.DataFrame(data_rows, columns=sheet_header)
        sheet_df.insert(0, "Sheet_State", sheet_name)
        frames.append(sheet_df)
    wb.close()

    if not frames:
        raise ValueError("No data rows found in any State/UT sheet.")
    data = pd.concat(frames, ignore_index=True)
    if "Year" not in data.columns:
        raise ValueError("State/UT sheets have no 'Year' column.")
    data["Year"] = data["Year"].astype(int)

    year_min, year_max = int(data["Year"].min()), int(data["Year"].max())
    label = f"{run_metadata.get('Scenario', source_name)} ({year_min}-{year_max})"
    return Scenario(
        label=label, source_name=source_name, run_metadata=run_metadata,
        indicator_metadata=indicator_metadata, data=data,
        year_min=year_min, year_max=year_max,
    )


def check_grid_spacing(data: pd.DataFrame, expected_deg: float = EXPECTED_GRID_SPACING_DEG) -> Optional[str]:
    """Returns a warning string if the data's actual grid spacing doesn't
    match the expected resolution, else None."""
    unique_lats = np.sort(data["Latitude"].unique())
    if len(unique_lats) < 2:
        return None
    actual = float(np.median(np.diff(unique_lats)))
    if not np.isclose(actual, expected_deg, atol=1e-6):
        return f"Median latitude spacing is {actual:.4f} deg, expected {expected_deg} deg."
    return None


def load_mapping(file_bytes: bytes) -> pd.DataFrame:
    """Load points_mapped.csv (District/State attribution per grid node) and
    standardise column names."""
    mapped = pd.read_csv(io.BytesIO(file_bytes), dtype={"DIST_LGD": str})
    mapped = mapped.rename(columns={
        "STATE_UT": "State", "DISTRICT": "District",
        "STATE_LGD": "State_ID", "DIST_LGD": "District_ID",
    })
    required = {"Latitude", "Longitude", "State", "District", "State_ID", "District_ID"}
    missing = required - set(mapped.columns)
    if missing:
        raise ValueError(f"Mapping file is missing expected columns: {sorted(missing)}")

    # Generic cleanup: strip stray whitespace/control characters some CSV
    # exports leave on text fields (confirmed present on 6 Assam district
    # names as trailing \r\n in this file) - always safe, no guessing involved.
    for col in ("State", "District"):
        mapped[col] = mapped[col].astype(str).str.strip()

    mapped["District"] = mapped["District"].replace(KNOWN_DISTRICT_NAME_FIXES)

    if mapped.duplicated(subset=["Latitude", "Longitude"]).any():
        mapped = mapped.drop_duplicates(subset=["Latitude", "Longitude"])
    return mapped


# Known source-data text corruption in points_mapped.csv: the Kannada long
# vowels (a-, i-, u-macron) in these district names were replaced with
# placeholder characters ('<', '\', '#') by whatever encoding/export step
# produced the file - confirmed by inspecting the raw district boundary
# shapefile's .dbf bytes directly (see the state/district mapping notebook).
# Two more (Dehradun, Birbhum) show the same pattern with '@'. All 21 are
# well-established, unambiguous Indian district names; corrected here rather
# than left broken. Please verify against your own reference list and flag
# this to whoever maintains the source shapefile.
KNOWN_DISTRICT_NAME_FIXES = {
    "B<galkot": "Bagalkot",
    "B\\dar": "Bidar",
    "Ball<ri": "Ballari",
    "Belag<vi": "Belagavi",
    "Bengal#ru (Rural)": "Bengaluru (Rural)",
    "Bengal#ru (Urban)": "Bengaluru (Urban)",
    "Ch<mar<janagara": "Chamarajanagara",
    "Chikkaball<pura": "Chikkaballapura",
    "Chikkamagal#ru": "Chikkamagaluru",
    "D<vanagere": "Davanagere",
    "Dh<rw<d": "Dharwad",
    "H<ssan": "Hassan",
    "H<veri": "Haveri",
    "Kol<ra": "Kolar",
    "Mys#ru": "Mysuru",
    "R<managara": "Ramanagara",
    "Raich#r": "Raichur",
    "Tumak#ru": "Tumakuru",
    "Y<dgir": "Yadgir",
    "DEHRAD@N": "DEHRADUN",
    "BIRBH@M": "BIRBHUM",
}


def coverage_gap(data: pd.DataFrame, mapped: pd.DataFrame) -> int:
    """Count of grid points in `data` with no matching row in `mapped`."""
    ind_points = data[["Latitude", "Longitude"]].drop_duplicates()
    merged = ind_points.merge(mapped[["Latitude", "Longitude"]], on=["Latitude", "Longitude"],
                               how="left", indicator=True)
    return int((merged["_merge"] == "left_only").sum())


def build_grid_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """One row per unique grid node with the list of workbook sheets it
    appears under (needed only to retrieve indicator rows for that node)."""
    return (
        data.groupby(["Latitude", "Longitude"])["Sheet_State"]
        .unique()
        .reset_index()
        .rename(columns={"Sheet_State": "Sheets"})
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance (km). Used instead of flat Euclidean
    degree-distance since 1 deg of longitude shrinks with latitude across
    India's ~6-37N span."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class GridMatch:
    lat: float
    lon: float
    sheets: list
    distance_km: float
    state: str
    district: str
    state_id: object
    district_id: object
    admin_match: str  # 'point' | 'overlap' | 'Unknown'


def nearest_grid_point(lat_in: float, lon_in: float, grid_lookup: pd.DataFrame,
                        admin_lookup: pd.DataFrame) -> GridMatch:
    """Find the nearest actual grid node to (lat_in, lon_in) and its District/
    State attribution."""
    d = haversine_km(lat_in, lon_in, grid_lookup["Latitude"].values, grid_lookup["Longitude"].values)
    i = int(np.argmin(d))
    row = grid_lookup.iloc[i]
    lat, lon = float(row["Latitude"]), float(row["Longitude"])
    try:
        admin = admin_lookup.loc[(lat, lon)]
        state, district, state_id, district_id, admin_match = (
            admin["State"], admin["District"], admin["State_ID"], admin["District_ID"], admin["match"],
        )
    except KeyError:
        state = district = state_id = district_id = "Unknown"
        admin_match = "Unknown"
    return GridMatch(lat, lon, list(row["Sheets"]), float(d[i]), state, district, state_id, district_id, admin_match)


def point_time_series(data: pd.DataFrame, lat: float, lon: float, indicator: str,
                       year_lo: int, year_hi: int) -> pd.DataFrame:
    """Yearly values of `indicator` at exactly (lat, lon), within [year_lo, year_hi].
    Border grid nodes are duplicated identically across two State sheets (verified
    in the notebook's Section 4); de-duplicate to one row per year."""
    subset = data[(data["Latitude"] == lat) & (data["Longitude"] == lon) &
                  (data["Year"].between(year_lo, year_hi))].sort_values("Year")
    return subset.drop_duplicates(subset=["Year"])[["Year", indicator]].reset_index(drop=True)


# --------------------------------------------------------------------------
# District-level spatial aggregation (via points_mapped.csv attribution)
# --------------------------------------------------------------------------

def list_states(mapped: pd.DataFrame) -> list[str]:
    return sorted(mapped["State"].dropna().unique().tolist())


def list_districts(mapped: pd.DataFrame, state: str) -> list[str]:
    return sorted(mapped.loc[mapped["State"] == state, "District"].dropna().unique().tolist())


def district_grid_values(data: pd.DataFrame, mapped: pd.DataFrame, state: str, district: str,
                          indicator: str, year: Optional[int] = None,
                          year_range: Optional[tuple[int, int]] = None) -> pd.DataFrame:
    """Grid-cell-level values of `indicator` for every node attributed to
    (state, district) in points_mapped.csv, either for a single `year` or
    averaged over `year_range` (inclusive). Returns one row per grid node:
    Latitude, Longitude, value, match (point/overlap fallback flag).

    Note on methodology: this attributes each 0.25 deg grid cell to a single
    district using its centroid's point-in-polygon result (with a nearest-
    district fallback for ~8% of coastal/border cells - see points_mapped.csv's
    'match' column). It is NOT an 'all-touched' polygon-raster clip (which
    would need the district boundary shapefile itself, not just this
    per-point attribution table) - a cell straddling two districts is
    counted for only one of them here. Swap in a true rioxarray/geopandas
    clip against the boundary shapefile if that distinction matters for your
    analysis.
    """
    if (year is None) == (year_range is None):
        raise ValueError("Pass exactly one of `year` or `year_range`.")

    nodes = mapped[(mapped["State"] == state) & (mapped["District"] == district)][
        ["Latitude", "Longitude", "match"]
    ]
    if nodes.empty:
        return nodes.assign(value=[])

    merged = data.merge(nodes, on=["Latitude", "Longitude"], how="inner")
    if year is not None:
        merged = merged[merged["Year"] == year]
        agg = merged.groupby(["Latitude", "Longitude", "match"], as_index=False)[indicator].mean()
    else:
        lo, hi = year_range
        merged = merged[merged["Year"].between(lo, hi)]
        agg = merged.groupby(["Latitude", "Longitude", "match"], as_index=False)[indicator].mean()
    return agg.rename(columns={indicator: "value"})


def district_time_series(data: pd.DataFrame, mapped: pd.DataFrame, state: str, district: str,
                          indicator: str, year_lo: int, year_hi: int) -> pd.DataFrame:
    """District-mean yearly time series: spatial mean of `indicator` across
    every grid node attributed to (state, district), for each year."""
    nodes = mapped[(mapped["State"] == state) & (mapped["District"] == district)][["Latitude", "Longitude"]]
    merged = data.merge(nodes, on=["Latitude", "Longitude"], how="inner")
    merged = merged[merged["Year"].between(year_lo, year_hi)]
    return (
        merged.groupby("Year", as_index=False)[indicator]
        .agg(mean="mean", min="min", max="max", n_cells="count")
    )


# --------------------------------------------------------------------------
# Scenario identity and ordering
# --------------------------------------------------------------------------
# Scenario labels are built from the workbook's own 'Scenario' metadata field
# ("Historical", "ssp245", ...) plus its year span, so they vary in case and
# formatting between files. Everything downstream (colour, ordering, "is this
# the historical run?") keys off the canonical short form derived here rather
# than off the display label, so a workbook labelled "SSP2-4.5", "ssp245" or
# "Projection ssp245" all resolve to the same scenario identity.

SSP_DISPLAY = {
    "historical": "Historical",
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp434": "SSP4-3.4",
    "ssp460": "SSP4-6.0",
    "ssp534os": "SSP5-3.4-OS",
    "ssp585": "SSP5-8.5",
}
# Radiative-forcing order (IPCC AR6 convention), historical first.
SCENARIO_ORDER = list(SSP_DISPLAY.keys())


def scenario_key(label: str) -> str:
    """Canonical short key for a scenario label: 'historical' or 'sspNNN'.

    Case/punctuation-insensitive, so "SSP2-4.5", "ssp245" and
    "Projection - ssp245 (2015-2037)" all collapse to 'ssp245'. An
    unrecognised label falls back to its own slug so it still gets a stable
    identity (colour, sort position) rather than colliding with others.
    """
    text = re.sub(r"\(.*?\)", " ", str(label))       # drop the "(1985-2014)" year span
    text = re.sub(r"\b\d{4}\b", " ", text)           # and any bare 4-digit year
    s = re.sub(r"[^a-z0-9]", "", text.lower())
    if "historical" in s or s.startswith("hist"):
        return "historical"
    m = re.search(r"ssp(\d{3})(os)?", s)             # ssp245, SSP2-4.5 -> ssp245; ssp534os
    if m:
        return f"ssp{m.group(1)}{m.group(2) or ''}"
    return s[:32] or "unknown"


def is_historical(label: str) -> bool:
    return scenario_key(label) == "historical"


def scenario_sort_key(label: str):
    """Sort scenarios historical-first, then by radiative forcing."""
    key = scenario_key(label)
    rank = SCENARIO_ORDER.index(key) if key in SCENARIO_ORDER else len(SCENARIO_ORDER)
    return (rank, str(label))


def sort_scenario_labels(labels) -> list[str]:
    return sorted(labels, key=scenario_sort_key)


def scenario_display_name(label: str) -> str:
    """Short, publication-ready scenario name ('Historical', 'SSP3-7.0') with
    the year span kept if the label carries one."""
    key = scenario_key(label)
    if key not in SSP_DISPLAY:
        return str(label)                            # unrecognised: show it verbatim
    m = re.search(r"\((\d{4}\s*-\s*\d{4})\)", str(label))
    return f"{SSP_DISPLAY[key]} ({m.group(1)})" if m else SSP_DISPLAY[key]


# --------------------------------------------------------------------------
# Multi-scenario point extraction (historical + SSPs on one axis)
# --------------------------------------------------------------------------

def multi_point_series(scenarios: dict, lat: float, lon: float, indicator: str,
                        year_lo: int, year_hi: int) -> pd.DataFrame:
    """Long-format yearly values at one grid node across several scenarios.

    `scenarios` is {label: Scenario}. Returns columns Scenario, Year, value,
    sorted historical-first then by forcing, then by year - the shape both the
    line and the bar renderers consume. Scenarios that do not carry
    `indicator`, or have no rows at this node in the window, are skipped
    silently (a scenario workbook may legitimately hold a different indicator
    set). NaNs are preserved: "not computable here" is information, not a row
    to drop.
    """
    frames = []
    for label in sort_scenario_labels(scenarios.keys()):
        sc = scenarios[label]
        if indicator not in sc.indicator_metadata or indicator not in sc.data.columns:
            continue
        ts = point_time_series(sc.data, lat, lon, indicator, year_lo, year_hi)
        if ts.empty:
            continue
        frames.append(ts.rename(columns={indicator: "value"}).assign(Scenario=label))
    if not frames:
        return pd.DataFrame({"Scenario": pd.Series(dtype=str),
                             "Year": pd.Series(dtype=int),
                             "value": pd.Series(dtype=float)})
    out = pd.concat(frames, ignore_index=True)[["Scenario", "Year", "value"]]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["Year"] = out["Year"].astype(int)
    out["_order"] = out["Scenario"].map(lambda s: scenario_sort_key(s)[0])
    return (out.sort_values(["_order", "Scenario", "Year"])
               .drop(columns="_order").reset_index(drop=True))


def splice_year(long_df: pd.DataFrame) -> Optional[int]:
    """The first projection year when historical and projected scenarios are
    shown together - i.e. where the record stops being observed-forced and
    starts being scenario-dependent. Returned so the figure can mark it;
    None when only one kind of run is on screen (nothing to mark).
    """
    if long_df.empty:
        return None
    labels = long_df["Scenario"].unique()
    hist = [l for l in labels if is_historical(l)]
    proj = [l for l in labels if not is_historical(l)]
    if not hist or not proj:
        return None
    proj_years = long_df.loc[long_df["Scenario"].isin(proj), "Year"]
    return int(proj_years.min()) if len(proj_years) else None


# --------------------------------------------------------------------------
# Temporal aggregation for bar charts
# --------------------------------------------------------------------------

def period_means(long_df: pd.DataFrame, period_len: int = 10) -> pd.DataFrame:
    """Mean of `value` per (Scenario, calendar-aligned period).

    Periods are anchored to calendar boundaries (`period_len * floor(year /
    period_len)`), so a 10-year window gives 1980-1989, 1990-1999, ... rather
    than windows that shift with whatever year range happens to be selected.
    That keeps the same bar comparable between two sessions and between
    scenarios, which a range-anchored binning does not.

    Returns Scenario, period_start, period_end, Period (label), mean, std,
    n_years, complete. `complete` is False where the selected year range
    covers only part of the period - those bars are means over fewer years
    and the caller should mark them rather than hide the fact.
    """
    cols = ["Scenario", "period_start", "period_end", "Period", "mean", "std", "n_years", "complete"]
    df = long_df.dropna(subset=["value"])
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    df = df.assign(period_start=(df["Year"] // period_len) * period_len)
    g = (df.groupby(["Scenario", "period_start"], as_index=False)
           .agg(mean=("value", "mean"), std=("value", "std"), n_years=("Year", "nunique")))
    g["period_end"] = g["period_start"] + period_len - 1
    g["Period"] = g["period_start"].astype(int).astype(str) + "-" + g["period_end"].astype(int).astype(str)
    g["complete"] = g["n_years"] >= period_len
    g["_order"] = g["Scenario"].map(lambda s: scenario_sort_key(s)[0])
    return (g.sort_values(["_order", "Scenario", "period_start"])
             .drop(columns="_order").reset_index(drop=True))[cols]


@dataclass
class Baseline:
    """Historical reference value at one grid node, for anomaly charts."""
    label: str          # scenario label the baseline came from
    year_lo: int
    year_hi: int
    mean: float
    n_years: int

    @property
    def period(self) -> str:
        return f"{self.year_lo}-{self.year_hi}"


def historical_baseline(scenarios: dict, lat: float, lon: float, indicator: str,
                         year_lo: Optional[int] = None,
                         year_hi: Optional[int] = None) -> Optional[Baseline]:
    """Mean of `indicator` at this node over the historical run's own period.

    Deliberately independent of the year range chosen for display: a climate
    anomaly is defined against a fixed reference period (here the historical
    workbook's full span, which is also the baseline used for the percentile
    indices - see the workbook's 'Baseline for Percentile Indices' field), not
    against whatever window the user happens to be looking at. Pass year_lo/
    year_hi only to override that reference period explicitly.

    Returns None when no historical workbook is loaded, or when the indicator
    has no valid year at this node - the caller must handle that rather than
    silently plotting anomalies against a NaN.
    """
    hist_labels = [l for l in scenarios if is_historical(l)]
    if not hist_labels:
        return None
    label = sort_scenario_labels(hist_labels)[0]
    sc = scenarios[label]
    if indicator not in sc.indicator_metadata or indicator not in sc.data.columns:
        return None
    lo = sc.year_min if year_lo is None else int(year_lo)
    hi = sc.year_max if year_hi is None else int(year_hi)
    ts = point_time_series(sc.data, lat, lon, indicator, lo, hi)
    vals = pd.to_numeric(ts[indicator], errors="coerce").dropna() if not ts.empty else pd.Series(dtype=float)
    if vals.empty:
        return None
    return Baseline(label=label, year_lo=lo, year_hi=hi,
                    mean=float(vals.mean()), n_years=int(vals.size))


# A percentage change is meaningless once the reference value is near zero
# (frost_days, cold_days etc. are legitimately 0 across most of India), so
# percent mode is suppressed below this absolute baseline magnitude instead of
# reporting a division blow-up as a climate signal.
PERCENT_BASELINE_FLOOR = 1e-6


def anomaly_vs_baseline(period_table: pd.DataFrame, baseline: Baseline,
                         mode: str = "absolute") -> pd.DataFrame:
    """Add an `anomaly` column: period mean minus the historical baseline mean
    (`mode='absolute'`, in the indicator's own units) or as a percentage of it
    (`mode='percent'`). Percent is returned as NaN when the baseline is
    effectively zero - see PERCENT_BASELINE_FLOOR."""
    if mode not in ("absolute", "percent"):
        raise ValueError("mode must be 'absolute' or 'percent'")
    t = period_table.copy()
    if t.empty:
        t["anomaly"] = pd.Series(dtype=float)
        return t
    diff = t["mean"].astype(float) - baseline.mean
    if mode == "percent":
        t["anomaly"] = np.nan if abs(baseline.mean) < PERCENT_BASELINE_FLOOR else diff / baseline.mean * 100.0
    else:
        t["anomaly"] = diff
    return t


def scenario_short_name(label: str) -> str:
    """Scenario name without its year span ('Historical', 'SSP3-7.0').

    Used for chart legends: the year span is already carried by the x-axis and
    by the app's own "Showing:" line, and repeating it four times inside a
    legend box makes the box wider than the panel at the larger text scales.
    """
    key = scenario_key(label)
    return SSP_DISPLAY.get(key, str(label))
