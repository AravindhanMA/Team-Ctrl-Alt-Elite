"""
Core data model for PS1 (Railway Track Access Optimisation).

Loads the 8 instance CSVs and builds the lookups needed to:
  - expand an activity's (start_location_id -> end_location_id) into the
    full ordered list of location_ids (tunnel sectors + platform sectors)
    it must book ("book-in" to "book-out"),
  - compute buffer footprints (exclusion zones) for Live / Non-live(Consist)
    work, including opposite-bound mirroring and the Live-only interchange
    crossover onto the other line's H01_H02 tunnel + platforms.

Run this file directly to sanity-check the footprint expansion against
the sample instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _p(name: str) -> str:
    return os.path.join(DATA_DIR, name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    lines: pd.DataFrame
    stations: pd.DataFrame
    sectors: pd.DataFrame
    location_supply: pd.DataFrame
    buffer_location: pd.DataFrame
    parameters: dict
    project_details: pd.DataFrame
    activity_details: pd.DataFrame

    # derived lookups, filled in by build_indexes()
    horizon_start: date = None
    horizon_weeks: int = 0

    # per (line, bound): ordered list of tunnel sector location_ids, by seq
    tunnel_seq: dict = field(default_factory=dict)
    # per (line, bound): ordered list of platform location_ids, by station seq
    platform_seq: dict = field(default_factory=dict)
    # sector_id -> (line, seq)   (seq is the SECTOR seq, not station seq)
    sector_seq_index: dict = field(default_factory=dict)
    # station_id, line -> station seq
    station_seq_index: dict = field(default_factory=dict)
    # location_id -> supply_capacity
    supply: dict = field(default_factory=dict)
    # nature_of_works -> (buffer_sectors, opposite_bound_required)
    buffer_rules: dict = field(default_factory=dict)


def load_instance(data_dir: str = DATA_DIR) -> Instance:
    global DATA_DIR
    DATA_DIR = data_dir

    lines = pd.read_csv(_p("01_LINES.csv"))
    stations = pd.read_csv(_p("02_STATIONS.csv"))
    sectors = pd.read_csv(_p("03_SECTORS.csv"))
    location_supply = pd.read_csv(_p("04_LOCATION_SUPPLY.csv"))
    buffer_location = pd.read_csv(_p("05_BUFFER_LOCATION.csv"))
    params_df = pd.read_csv(_p("06_PARAMETERS.csv"))
    parameters = dict(zip(params_df["key"], params_df["value"]))
    project_details = pd.read_csv(_p("07_PROJECT_DETAILS.csv"))
    activity_details = pd.read_csv(_p("08_ACTIVITY_DETAILS.csv"))

    inst = Instance(
        lines=lines,
        stations=stations,
        sectors=sectors,
        location_supply=location_supply,
        buffer_location=buffer_location,
        parameters=parameters,
        project_details=project_details,
        activity_details=activity_details,
    )
    build_indexes(inst)
    return inst


def build_indexes(inst: Instance) -> None:
    y, m, d = (int(x) for x in str(inst.parameters["horizon_start"]).split("-"))
    inst.horizon_start = date(y, m, d)
    inst.horizon_weeks = int(inst.parameters["horizon_weeks"])

    # station seq index: (line, station_id) -> seq
    for _, row in inst.stations.iterrows():
        inst.station_seq_index[(row["line_code"], row["station_id"])] = row["seq"]

    # sector seq index: sector_id -> (line, seq, from_station, to_station)
    for _, row in inst.sectors.iterrows():
        inst.sector_seq_index[row["sector_id"]] = {
            "line": row["line_code"],
            "seq": row["seq"],
            "from": row["from_station_id"],
            "to": row["to_station_id"],
        }

    # tunnel_seq[(line,bound)] = list of (seq, location_id) sorted by seq
    for line in inst.lines["line_code"]:
        for bound in ("EB", "WB"):
            secs = inst.sectors[inst.sectors["line_code"] == line].sort_values("seq")
            tseq = [(int(r["seq"]), f"{r['sector_id']}:{bound}") for _, r in secs.iterrows()]
            inst.tunnel_seq[(line, bound)] = tseq

            stas = inst.stations[inst.stations["line_code"] == line].sort_values("seq")
            pseq = [(int(r["seq"]), f"PLAT:{line}:{r['station_id']}:{bound}") for _, r in stas.iterrows()]
            inst.platform_seq[(line, bound)] = pseq

    # supply
    for _, row in inst.location_supply.iterrows():
        inst.supply[row["location_id"]] = int(row["supply_capacity"])

    # buffer rules
    for _, row in inst.buffer_location.iterrows():
        inst.buffer_rules[row["nature_of_works"]] = {
            "buffer_sectors": int(row["up_to_buffer_sectors"]),
            "opposite_bound_required": bool(row["opposite_bound_required"]),
        }


# ---------------------------------------------------------------------------
# Location_id parsing helpers
# ---------------------------------------------------------------------------

def parse_tunnel_location(location_id: str):
    """'SEC:ALP:S01_S02:EB' -> (sector_id='SEC:ALP:S01_S02', bound='EB')"""
    parts = location_id.split(":")
    bound = parts[-1]
    sector_id = ":".join(parts[:-1])
    return sector_id, bound


_DIRECTION_NAMES = {"EB": "Eastbound", "WB": "Westbound"}


def humanize_location(inst: "Instance", location_id: str) -> str:
    """UI-display only -- never used for the submission CSVs, which must
    keep the raw location_id format.

    'SEC:BET:H02_S15:EB' -> 'Line Beta, between H02 and S15 (Eastbound)'
    'PLAT:ALP:S03:WB'    -> 'Line Alpha, S03 platform (Westbound)'

    Falls back to the raw ID for anything that doesn't match the expected
    shape, so this never hides or breaks on unexpected data.
    """
    line_names = dict(zip(inst.lines["line_code"], inst.lines["line_name"]))
    parts = location_id.split(":")
    try:
        if parts[0] == "SEC" and len(parts) == 4:
            _, line, span, bound = parts
            frm, to = span.split("_", 1)
            return f"{line_names.get(line, line)}, between {frm} and {to} ({_DIRECTION_NAMES.get(bound, bound)})"
        elif parts[0] == "PLAT" and len(parts) == 4:
            _, line, station, bound = parts
            return f"{line_names.get(line, line)}, {station} platform ({_DIRECTION_NAMES.get(bound, bound)})"
    except Exception:
        pass
    return location_id


# ---------------------------------------------------------------------------
# Footprint expansion: activity's start/end location -> full location list
# ---------------------------------------------------------------------------

def expand_footprint(inst: Instance, start_location_id: str, end_location_id: str):
    """
    Returns dict with:
      line, bound,
      tunnel_locations: ordered list of tunnel sector location_ids booked
      platform_locations: ordered list of platform location_ids booked
      all_locations: tunnel + platform combined (booking footprint)
      seq_lo, seq_hi: the sector-seq range spanned (inclusive)
    Raises ValueError if start/end aren't on the same line+bound, or
    if the sector isn't found.
    """
    start_sector_id, start_bound = parse_tunnel_location(start_location_id)
    end_sector_id, end_bound = parse_tunnel_location(end_location_id)

    if start_bound != end_bound:
        raise ValueError(
            f"start/end bound mismatch: {start_location_id} vs {end_location_id}"
        )
    bound = start_bound

    start_info = inst.sector_seq_index.get(start_sector_id)
    end_info = inst.sector_seq_index.get(end_sector_id)
    if start_info is None or end_info is None:
        raise ValueError(f"unknown sector in {start_location_id} / {end_location_id}")
    if start_info["line"] != end_info["line"]:
        raise ValueError(
            f"start/end on different lines: {start_location_id} vs {end_location_id}"
        )
    line = start_info["line"]

    seq_lo, seq_hi = sorted([start_info["seq"], end_info["seq"]])

    tseq = inst.tunnel_seq[(line, bound)]
    tunnel_locations = [loc for (seq, loc) in tseq if seq_lo <= seq <= seq_hi]

    # station seq range: union of the endpoints of every sector in range
    station_seqs = set()
    for seq, loc in tseq:
        if seq_lo <= seq <= seq_hi:
            sector_id, _ = parse_tunnel_location(loc)
            info = inst.sector_seq_index[sector_id]
            station_seqs.add(inst.station_seq_index[(line, info["from"])])
            station_seqs.add(inst.station_seq_index[(line, info["to"])])
    st_lo, st_hi = min(station_seqs), max(station_seqs)

    pseq = inst.platform_seq[(line, bound)]
    platform_locations = [loc for (seq, loc) in pseq if st_lo <= seq <= st_hi]

    return {
        "line": line,
        "bound": bound,
        "tunnel_locations": tunnel_locations,
        "platform_locations": platform_locations,
        "all_locations": tunnel_locations + platform_locations,
        "seq_lo": seq_lo,
        "seq_hi": seq_hi,
    }


def week_number(inst: Instance, d: date) -> int:
    """1-indexed week number relative to horizon_start."""
    return ((d - inst.horizon_start).days // 7) + 1


def week_start_date(inst: Instance, week: int) -> date:
    return inst.horizon_start + timedelta(days=7 * (week - 1))


# ---------------------------------------------------------------------------
# Self-test / sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inst = load_instance()
    print(f"Horizon: {inst.horizon_start}, {inst.horizon_weeks} weeks")
    print(f"Locations with supply data: {len(inst.supply)}")
    print(f"Activities: {len(inst.activity_details)}")
    print()

    # test a few real activities, including single-sector and multi-sector,
    # and one that spans through an interchange
    test_ids = ["A001", "A003", "A007", "A017", "A012"]
    for aid in test_ids:
        row = inst.activity_details[inst.activity_details["activity_id"] == aid].iloc[0]
        fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
        print(f"{aid}: {row['start_location_id']} -> {row['end_location_id']}")
        print(f"   line={fp['line']} bound={fp['bound']}")
        print(f"   tunnel ({len(fp['tunnel_locations'])}): {fp['tunnel_locations']}")
        print(f"   platform ({len(fp['platform_locations'])}): {fp['platform_locations']}")
        print()

    # validate footprint expansion for ALL activities doesn't error
    errors = 0
    for _, row in inst.activity_details.iterrows():
        try:
            fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
            for loc in fp["all_locations"]:
                if loc not in inst.supply:
                    print(f"WARNING: {row['activity_id']} footprint location {loc} not in LOCATION_SUPPLY")
        except Exception as e:
            print(f"ERROR on {row['activity_id']}: {e}")
            errors += 1
    print(f"Footprint expansion tested on all {len(inst.activity_details)} activities, {errors} errors")
