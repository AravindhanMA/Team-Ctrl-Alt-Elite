"""
Closure + buffer footprint for a single possession, given its nature_of_works.

Two zones matter, per §2.4:
  - CLOSURE: the exact locations the activity occupies. No other activity
    (co-sharing partners excepted) may use these on that night, regardless
    of nature_of_works.
  - BUFFER: extra sectors ahead/behind the closure, on the SAME bound, that
    only apply to OTHER Live/Non-live(Consist) work (Non-live(Others) has
    no buffer, and doesn't need to respect others' buffers when it comes to
    picking a sector -- but see note below). We model this as extending the
    reserved sector-seq range for collision-checking purposes.

ASSUMPTIONS made explicit here (the spec is not 100% literal on these, so
flag them in your write-up):
  1. "up_to_buffer_sectors" extends the TUNNEL sector-seq range only
     (not platforms) -- physically buffers are about clear track ahead of
     worksite. Platforms are governed purely by their own capacity.
  2. For Live's cross-line interchange crossover, we close the OTHER line's
     H01_H02 tunnel sector on BOTH bounds, plus all 4 of that line's H01/H02
     platforms (EB+WB) -- a traction power cut at the interchange is treated
     as bidirectional and affects both lines fully, since 750V is physically
     shared infrastructure at the hub, not bound-specific.
  3. Buffer zones do NOT extend across the interchange onto the other line;
     they only extend within the same line+bound's sector-seq chain.
"""
from __future__ import annotations

from data_model import Instance, parse_tunnel_location, expand_footprint


def other_line(inst: Instance, line: str) -> str:
    """Derived from the instance's own line list, not hardcoded names --
    works for any 2-line topology, not just literally ALP/BET."""
    others = [l for l in inst.lines["line_code"] if l != line]
    if len(others) != 1:
        raise ValueError(
            f"other_line() assumes exactly 2 lines (found {list(inst.lines['line_code'])}); "
            "the Live interchange-crossover rule as specified only defines behavior for a "
            "2-line network, so a 3+ line instance would need this rule re-specified."
        )
    return others[0]


def interchange_sector_ids(inst: Instance, line: str) -> set:
    """Sector(s) on this line connecting two interchange (is_interchange=1)
    stations -- derived from 02_STATIONS.csv rather than hardcoded 'H01_H02',
    so this still works if a different instance names its interchange hub
    stations something else."""
    interchange_stations = set(
        inst.stations[(inst.stations["line_code"] == line) & (inst.stations["is_interchange"] == 1)]["station_id"]
    )
    result = set()
    for _, row in inst.sectors[inst.sectors["line_code"] == line].iterrows():
        if row["from_station_id"] in interchange_stations and row["to_station_id"] in interchange_stations:
            result.add(row["sector_id"])
    return result


def closure_and_buffer(inst: Instance, nature_of_works: str, footprint: dict):
    """
    footprint: result of expand_footprint() for this activity.
    Returns dict:
      closure: set of location_ids that are fully occupied (hard exclusive,
               except for co-sharing partners in the same co_share_group)
      buffer_extra: set of EXTRA tunnel location_ids (beyond closure) that
               count as reserved for buffer-collision purposes against other
               Live/Non-live(Consist) work
      opposite_bound_closure: set of location_ids mirrored onto the opposite
               bound (Live only) -- these are full closures, same as `closure`
      interchange_closure: set of location_ids on the OTHER line, closed due
               to Live's interchange crossover (only if footprint touches
               H01_H02 on this line)
    """
    rule = inst.buffer_rules[nature_of_works]
    buffer_n = rule["buffer_sectors"]
    mirror_opposite_bound = rule["opposite_bound_required"]

    line = footprint["line"]
    bound = footprint["bound"]
    seq_lo, seq_hi = footprint["seq_lo"], footprint["seq_hi"]

    closure = set(footprint["all_locations"])

    # --- buffer: extend tunnel seq range on the SAME bound ---
    buffer_extra = set()
    if buffer_n > 0:
        tseq = inst.tunnel_seq[(line, bound)]
        ext_lo, ext_hi = seq_lo - buffer_n, seq_hi + buffer_n
        for seq, loc in tseq:
            if ext_lo <= seq <= ext_hi and loc not in closure:
                buffer_extra.add(loc)

    # --- opposite bound mirror (Live only, per 05_BUFFER_LOCATION flag) ---
    opposite_bound_closure = set()
    opposite_bound_buffer = set()
    if mirror_opposite_bound:
        opp_bound = "WB" if bound == "EB" else "EB"
        opp_tseq = inst.tunnel_seq[(line, opp_bound)]
        opp_pseq = inst.platform_seq[(line, opp_bound)]
        for seq, loc in opp_tseq:
            if seq_lo <= seq <= seq_hi:
                opposite_bound_closure.add(loc)
        # mirror the platform range too (same station range)
        station_seqs = set()
        for seq, loc in inst.tunnel_seq[(line, bound)]:
            if seq_lo <= seq <= seq_hi:
                sector_id, _ = parse_tunnel_location(loc)
                info = inst.sector_seq_index[sector_id]
                station_seqs.add(inst.station_seq_index[(line, info["from"])])
                station_seqs.add(inst.station_seq_index[(line, info["to"])])
        if station_seqs:
            st_lo, st_hi = min(station_seqs), max(station_seqs)
            for seq, loc in opp_pseq:
                if st_lo <= seq <= st_hi:
                    opposite_bound_closure.add(loc)
        if buffer_n > 0:
            ext_lo, ext_hi = seq_lo - buffer_n, seq_hi + buffer_n
            for seq, loc in opp_tseq:
                if ext_lo <= seq <= ext_hi and loc not in opposite_bound_closure:
                    opposite_bound_buffer.add(loc)

    # --- interchange crossover (Live only) ---
    interchange_closure = set()
    if nature_of_works == "Live":
        hub_sectors = interchange_sector_ids(inst, line)
        touches_hub = any(
            parse_tunnel_location(loc)[0] in hub_sectors for loc in footprint["tunnel_locations"]
        ) or any(
            parse_tunnel_location(loc)[0] in hub_sectors for loc in opposite_bound_closure
        )
        if touches_hub:
            oline = other_line(inst, line)
            oline_hub_sectors = interchange_sector_ids(inst, oline)
            oline_hub_stations = set()
            for sector_id in oline_hub_sectors:
                info = inst.sector_seq_index[sector_id]
                oline_hub_stations.add(info["from"])
                oline_hub_stations.add(info["to"])
            for b in ("EB", "WB"):
                for sector_id in oline_hub_sectors:
                    interchange_closure.add(f"{sector_id}:{b}")
                for station_id in oline_hub_stations:
                    interchange_closure.add(f"PLAT:{oline}:{station_id}:{b}")

    return {
        "closure": closure,
        "buffer_extra": buffer_extra,
        "opposite_bound_closure": opposite_bound_closure,
        "opposite_bound_buffer": opposite_bound_buffer,
        "interchange_closure": interchange_closure,
    }


if __name__ == "__main__":
    from data_model import load_instance

    inst = load_instance()
    proj = inst.project_details.set_index("contract_number")

    for aid in ["A074", "A075", "A001", "A007"]:
        row = inst.activity_details[inst.activity_details["activity_id"] == aid].iloc[0]
        nature = proj.loc[row["contract_number"], "nature_of_activity"]
        fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
        cb = closure_and_buffer(inst, nature, fp)
        print(f"{aid} ({nature}, {fp['line']}-{fp['bound']}):")
        print(f"  closure ({len(cb['closure'])}): {sorted(cb['closure'])}")
        print(f"  buffer_extra ({len(cb['buffer_extra'])}): {sorted(cb['buffer_extra'])}")
        print(f"  opposite_bound_closure ({len(cb['opposite_bound_closure'])}): {sorted(cb['opposite_bound_closure'])}")
        print(f"  interchange_closure ({len(cb['interchange_closure'])}): {sorted(cb['interchange_closure'])}")
        print()
