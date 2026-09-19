"""
Independent self-validator. Re-derives everything from the raw CSV output
(not from the scheduler's internal state) so it acts as a genuine second
check, not just an echo of the scheduler's own bookkeeping.

Checks (Scenario A):
  - workload conservation: every activity's placed nights sum >= total_accesses
  - planned start date respected
  - predecessor precedence (finish-to-start, strictly later week)
  - location capacity never exceeded (per location, per week)
  - legal mix respected (<=1 PM alone / <=1PC+<=3C / <=4C) per location-week
  - weekly allocation cap (rule 7) per contract+type+week
  - workfront cap (rule 8) per contract+type+week+access_night
  - ECLO forbidden (Scenario A only)
"""
from __future__ import annotations

import math
import pandas as pd

from data_model import Instance, load_instance, expand_footprint, week_number
from buffers import closure_and_buffer


def validate(inst: Instance, access_df: pd.DataFrame, occ_df: pd.DataFrame, scenario: str):
    violations = []
    proj = inst.project_details.set_index("contract_number")
    act = inst.activity_details.set_index("activity_id")

    # --- rebuild per-activity footprint/closure info ---
    footprints = {}
    for aid, row in act.iterrows():
        contract = proj.loc[row["contract_number"]]
        fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
        cb = closure_and_buffer(inst, contract["nature_of_activity"], fp)
        closure = cb["closure"] | cb["opposite_bound_closure"] | cb["interchange_closure"]
        buffer_locs = cb["buffer_extra"] | cb["opposite_bound_buffer"]
        footprints[aid] = {"closure": closure, "buffer": buffer_locs, "contract": contract, "row": row}

    # --- 1. workload conservation ---
    for aid, row in act.iterrows():
        a = access_df[access_df["activity_id"] == aid]
        units = a.apply(lambda r: 1.5 if r["eclo"] == 1 else 1.0, axis=1).sum()
        if units < row["total_accesses"] - 1e-9:
            violations.append({"rule": "workload", "severity": "hard",
                                "detail": f"{aid}: only {units} of {row['total_accesses']} accesses scheduled"})

    # --- 2. planned start date ---
    for aid, row in act.iterrows():
        a = access_df[access_df["activity_id"] == aid]
        if a.empty:
            continue
        earliest_allowed = week_number(inst, pd.Timestamp(row["planned_start_date"]).date())
        first_week = a["week"].min()
        if first_week < earliest_allowed:
            violations.append({"rule": "planned_date", "severity": "hard",
                                "detail": f"{aid}: starts week {first_week} before planned week {earliest_allowed}"})

    # --- 3. predecessor precedence ---
    for aid, row in act.iterrows():
        pred = row.get("predecessor_activity_id")
        if pd.isna(pred) or pred == "":
            continue
        pred_a = access_df[access_df["activity_id"] == pred]
        succ_a = access_df[access_df["activity_id"] == aid]
        if pred_a.empty or succ_a.empty:
            continue
        pred_finish = pred_a["week"].max()
        succ_start = succ_a["week"].min()
        if succ_start <= pred_finish:
            violations.append({"rule": "predecessor", "severity": "hard",
                                "detail": f"{aid}: starts week {succ_start}, predecessor {pred} finishes week {pred_finish}"})

    # --- 4/5. capacity + legal mix, per (location, week) ---
    # capacity tolerance depends on scenario: A = 0 tolerance, B = unlimited
    # (never hard-fails on capacity), C = up to 1 excess per location-week.
    # Per mentor guidance: in Scenario B, "Strict Schedule, Flexible Supply"
    # is read to cover ALL location-based packing rules (not just raw
    # numeric capacity) as flexible -- so legal-mix (PM-alone / <=1 PC) is
    # also NOT hard-failed in B, only in A and C, matching the scheduler's
    # schedule_scenario_b_exact, which no longer hard-constrains this for B.
    merged = occ_df.merge(act[["contract_number"]], left_on="activity_id", right_index=True)
    merged = merged.merge(proj[["access_type"]], left_on="contract_number", right_index=True)
    for (loc, week), grp in merged.groupby(["location_id", "week"]):
        cap = inst.supply.get(loc, 0)
        n_activities = grp["activity_id"].nunique()
        if scenario == "A":
            allowed = cap
        elif scenario == "C":
            allowed = cap + 1
        else:  # B: capacity is soft-scored only, never hard-fails
            allowed = 10**9
        if n_activities > allowed:
            violations.append({"rule": "capacity", "severity": "hard",
                                "detail": f"wk{week}: {loc} has {n_activities} activities, capacity {cap}"})
        if scenario != "B":
            types = grp.drop_duplicates("activity_id")["access_type"].tolist()
            n_pm = types.count("PM")
            n_pc = types.count("PC")
            if n_pm > 0 and len(types) > 1:
                violations.append({"rule": "legal_mix", "severity": "hard",
                                    "detail": f"wk{week}: {loc} has PM sharing with others: {types}"})
            if n_pc > 1:
                violations.append({"rule": "legal_mix", "severity": "hard",
                                    "detail": f"wk{week}: {loc} has multiple PC: {types}"})

    # --- 5b. buffer collisions (a genuine gap this validator didn't check
    #     before -- closure/buffer zones between DIFFERENT activities must
    #     never overlap, regardless of contract; only an EXACT matching
    #     location+week co-occupancy, already covered by the capacity/
    #     legal-mix check above, is exempt from this). Per mentor guidance,
    #     also NOT hard-failed for Scenario B, for the same reason as
    #     legal-mix above -- only A and C treat this as hard. ---
    if scenario != "B":
        active_by_week = {}
        for week, grp in access_df.groupby("week"):
            active_by_week[week] = grp["activity_id"].unique().tolist()
        for week, active_aids in active_by_week.items():
            for i in range(len(active_aids)):
                for j in range(i + 1, len(active_aids)):
                    a1, a2 = active_aids[i], active_aids[j]
                    if a1 not in footprints or a2 not in footprints:
                        continue
                    f1, f2 = footprints[a1], footprints[a2]
                    collision = bool(
                        (f1["closure"] & f2["buffer"]) or
                        (f1["buffer"] & f2["closure"]) or
                        (f1["buffer"] & f2["buffer"])
                    )
                    if collision:
                        violations.append({"rule": "buffer", "severity": "hard",
                                            "detail": f"wk{week}: {a1} and {a2} have overlapping buffer/closure zones"})

    # --- 6. weekly allocation cap (rule 7) ---
    access_merged = access_df.merge(act[["contract_number"]], left_on="activity_id", right_index=True)
    access_merged = access_merged.merge(proj[["number_of_maximum_access_per_week", "number_of_workfronts"]],
                                          left_on="contract_number", right_index=True)
    for (contract, week), grp in access_merged.groupby(["contract_number", "week"]):
        cap = grp["number_of_maximum_access_per_week"].iloc[0]
        n_nights = grp["access_night"].nunique()
        if n_nights > cap:
            violations.append({"rule": "weekly_allocation", "severity": "hard",
                                "detail": f"{contract} wk{week}: uses {n_nights} distinct access_night values, cap {cap}"})

    # --- 7. workfront cap (rule 8) ---
    for (contract, week, night), grp in access_merged.groupby(["contract_number", "week", "access_night"]):
        wf = grp["number_of_workfronts"].iloc[0]
        n_act = grp["activity_id"].nunique()
        if n_act > wf:
            violations.append({"rule": "workfront", "severity": "hard",
                                "detail": f"{contract} wk{week} night{night}: {n_act} concurrent activities, workfront cap {wf}"})

    # --- 8. ECLO forbidden in Scenario A ---
    if scenario == "A":
        eclo_rows = access_df[access_df["eclo"] == 1]
        if len(eclo_rows) > 0:
            violations.append({"rule": "eclo", "severity": "hard",
                                "detail": f"{len(eclo_rows)} ECLO nights used in Scenario A (forbidden)"})

    # --- 9. Scenario B: planned_completion_date is rigid, never breached ---
    if scenario == "B":
        for contract_number, contract in proj.iterrows():
            contract_activity_ids = act[act["contract_number"] == contract_number].index
            weeks = access_df[access_df["activity_id"].isin(contract_activity_ids)]["week"]
            if weeks.empty:
                continue
            finish_week = weeks.max()
            deadline_week = week_number(inst, pd.Timestamp(contract["planned_completion_date"]).date())
            if finish_week > deadline_week:
                violations.append({"rule": "planned_date", "severity": "hard",
                                    "detail": f"{contract_number}: finishes week {finish_week}, "
                                              f"planned_completion_date is week {deadline_week}"})

    # --- 10. Scenario C: ECLO continuity window (<=2 weeks per line) ---
    if scenario == "C":
        act_line = {}
        for aid, row in act.iterrows():
            fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
            act_line[aid] = fp["line"]
        eclo_rows = access_df[access_df["eclo"] == 1].copy()
        eclo_rows["line"] = eclo_rows["activity_id"].map(act_line)
        for line, grp in eclo_rows.groupby("line"):
            span = grp["week"].max() - grp["week"].min()
            if span > 1:  # more than a 2-calendar-week span
                violations.append({"rule": "eclo_window", "severity": "hard",
                                    "detail": f"{line}: ECLO nights span weeks "
                                              f"{grp['week'].min()}-{grp['week'].max()} (>2 weeks)"})

    return violations


if __name__ == "__main__":
    from scheduler import schedule_scenario_a, build_output_frames

    inst = load_instance()
    state, info = schedule_scenario_a(inst)
    access_df, occ_df, results_df = build_output_frames(inst, state, info, "A")

    violations = validate(inst, access_df, occ_df, "A")
    print(f"Hard violations found: {len(violations)}")
    for v in violations[:30]:
        print(f"  [{v['rule']}] {v['detail']}")
    if len(violations) > 30:
        print(f"  ... and {len(violations) - 30} more")

    print()
    print("FEASIBLE" if not violations else "NOT FEASIBLE")
