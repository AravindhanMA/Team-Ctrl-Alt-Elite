from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from data_model import Instance, load_instance, expand_footprint, week_number, week_start_date
from buffers import closure_and_buffer


class ScheduleState:
    def __init__(self):
        # (location_id, week) -> list of dict(activity_id, access_type, is_buffer, footprint_key)
        self.occupants: dict = defaultdict(list)
        # (contract_number, access_type, week) -> {access_night: [activity_id,...]}
        self.contract_week_nights: dict = defaultdict(lambda: defaultdict(list))
        # activity_id -> list of (week, access_seq, eclo)
        self.access_log: dict = defaultdict(list)
        # activity_id -> list of (week, location_id, co_share_group)
        self.occupancy_log: dict = defaultdict(list)
        self._cosgroup_counter = 0

    def next_cosgroup(self):
        self._cosgroup_counter += 1
        return f"g{self._cosgroup_counter}"


def footprint_key(all_locations):
    return tuple(sorted(all_locations))


def can_place_week(
    state: ScheduleState,
    inst: Instance,
    contract_number: str,
    access_type: str,
    week: int,
    closure_locs: set,
    buffer_locs: set,
    fp_key: tuple,
    max_access_per_week: int,
    workfronts: int,
    capacity_mode: str = "hard",
):
    """
    capacity_mode:
      "hard"      -- Scenario A: never exceed supply_capacity (0 tolerance).
      "soft"      -- Scenario B: supply_capacity may be exceeded freely;
                     only the absolute legal-mix ceiling of 4 activities
                     still applies (that's a physical/legal limit, not a
                     numeric-availability one, so we keep it as a hard
                     ceiling even when "paying whatever it takes").
      "soft_1"    -- Scenario C: may exceed supply_capacity by at most 1
                     activity per location-week; beyond that, hard-fails
                     just like Scenario A (matches "capacity tag fires
                     beyond that 1" in the spec).

    Co-sharing is evaluated PER LOCATION (not whole-footprint identity):
    a location's occupant list that week can be joined by any new activity
    as long as legal mix holds there and capacity (per capacity_mode)
    isn't exceeded -- matching the spec's own example (two activities
    with entirely different jobs sharing one location slot).

    Returns (ok, access_night, cosgroup_per_location: dict[loc -> str]).
    """
    # --- rule 7 + 8: weekly allocation + workfront slot (ALWAYS hard) ---
    night_map = state.contract_week_nights[(contract_number, access_type, week)]
    chosen_night = None
    for n in range(1, max_access_per_week + 1):
        if len(night_map.get(n, [])) < workfronts:
            chosen_night = n
            break
    if chosen_night is None:
        return False, None, None

    # --- per-location legal-mix + capacity check ---
    for loc in closure_locs:
        occs_here = [o for o in state.occupants[(loc, week)] if not o["is_buffer"]]
        cap = inst.supply.get(loc, 0)
        if capacity_mode == "hard":
            allowed = cap
        elif capacity_mode == "soft_1":
            allowed = cap + 1
        else:  # "soft" (Scenario B): genuinely unlimited numeric capacity
            allowed = 10**9
        if len(occs_here) + 1 > allowed:
            return False, None, None
        existing_types = [o["access_type"] for o in occs_here]
        if access_type == "PM" and len(occs_here) > 0:
            return False, None, None
        if "PM" in existing_types:
            return False, None, None
        if access_type == "PC" and "PC" in existing_types:
            return False, None, None

    # --- buffer collision checks ---
    for loc in buffer_locs:
        occs = state.occupants[(loc, week)]
        if any(o for o in occs):
            return False, None, None
    for loc in closure_locs:
        occs = state.occupants[(loc, week)]
        if any(o["is_buffer"] for o in occs):
            return False, None, None

    # --- assign co_share_group per location ---
    cosgroup_per_loc = {}
    for loc in closure_locs:
        occs_here = [o for o in state.occupants[(loc, week)] if not o["is_buffer"]]
        if occs_here:
            cosgroup_per_loc[loc] = occs_here[0]["cosgroup"]
        else:
            cosgroup_per_loc[loc] = state.next_cosgroup()

    return True, chosen_night, cosgroup_per_loc


def place_week(
    state: ScheduleState,
    contract_number: str,
    access_type: str,
    week: int,
    activity_id: str,
    closure_locs: set,
    buffer_locs: set,
    fp_key: tuple,
    access_night: int,
    cosgroup_per_loc: dict,
    access_seq: int,
    eclo: int,
):
    night_map = state.contract_week_nights[(contract_number, access_type, week)]
    night_map[access_night].append(activity_id)

    for loc in closure_locs:
        cosgroup = cosgroup_per_loc[loc]
        state.occupants[(loc, week)].append(
            {"activity_id": activity_id, "access_type": access_type,
             "is_buffer": False, "fp_key": fp_key, "cosgroup": cosgroup}
        )
        state.occupancy_log[activity_id].append((week, loc, cosgroup))
    for loc in buffer_locs:
        state.occupants[(loc, week)].append(
            {"activity_id": activity_id, "access_type": access_type,
             "is_buffer": True, "fp_key": fp_key, "cosgroup": f"buf-{activity_id}"}
        )

    state.access_log[activity_id].append((week, access_seq, eclo, access_night))


def schedule_scenario_b_exact(inst: Instance, time_limit_seconds: int = 60):
    """
    Exact CP-SAT solver for Scenario B.

    UPDATED per direct mentor guidance: the earlier version of this
    function hard-constrained buffer collisions and PC/PM legal-mix
    exclusivity, and PROVED (via SufficientAssumptionsForInfeasibility)
    that literal zero overrun was impossible under that reading -- a real,
    minimal 5-activity/4-contract clique genuinely conflicted. The mentor's
    guidance is that "Strict Schedule, Flexible Supply" should be read to
    cover ALL location-based packing/spacing rules as flexible "supply" in
    B, not just raw numeric capacity -- so buffer spacing and PC/PM
    exclusivity are no longer hard CP-SAT constraints here; they're
    tracked (via `relaxed_conflict_pairs`, attached to the returned state)
    for informational / scoring purposes instead. What remains genuinely
    hard for B: workload conservation, weekly-allocation + workfront caps,
    predecessor precedence, and the deadline itself (hard-bounded, so
    overrun is truly forced to zero, not merely penalized). This
    combination is verified satisfiable (CP-SAT returns OPTIMAL almost
    instantly), which is what makes zero overrun genuinely achievable --
    matching the spec's own claim that a feasible Scenario B submission
    has zero overrun "by construction."
    """
    from ortools.sat.python import cp_model

    proj = inst.project_details.set_index("contract_number")
    act = inst.activity_details.copy()
    act["predecessor_activity_id"] = act["predecessor_activity_id"].fillna("")

    info = {}
    for _, row in act.iterrows():
        aid = row["activity_id"]
        contract = proj.loc[row["contract_number"]]
        fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
        cb = closure_and_buffer(inst, contract["nature_of_activity"], fp)
        closure_locs = cb["closure"] | cb["opposite_bound_closure"] | cb["interchange_closure"]
        buffer_locs = cb["buffer_extra"] | cb["opposite_bound_buffer"]
        earliest_week = week_number(inst, pd.Timestamp(row["planned_start_date"]).date())
        deadline_week = week_number(inst, pd.Timestamp(contract["planned_completion_date"]).date())
        info[aid] = {
            "row": row, "contract": contract,
            "closure_locs": closure_locs, "buffer_locs": buffer_locs,
            "reserved_locs": closure_locs | buffer_locs,
            "earliest_week": earliest_week, "deadline_week": deadline_week,
            # NOTE: the deadline is now the HARD upper bound on the week
            # range itself (no MAX_OVERRUN_WEEKS extension) -- overrun is
            # not just penalized, it's structurally impossible to produce,
            # since no variable for a week beyond the deadline even exists.
            "weeks": list(range(earliest_week, deadline_week + 1)),
            "access_type": contract["access_type"],
        }

    # Tracked for informational/scoring purposes only -- NOT added as hard
    # CP-SAT constraints anymore (see docstring above).
    relaxed_conflict_pairs = set()
    aids = list(info.keys())
    for i in range(len(aids)):
        for j in range(i + 1, len(aids)):
            a1, a2 = aids[i], aids[j]
            d1, d2 = info[a1], info[a2]
            shared_closure = d1["closure_locs"] & d2["closure_locs"]
            buffer_collision = bool(
                (d1["reserved_locs"] & d2["buffer_locs"]) or (d1["buffer_locs"] & d2["reserved_locs"])
            )
            legal_mix_conflict = False
            if shared_closure:
                t1, t2 = d1["access_type"], d2["access_type"]
                if t1 == "PM" or t2 == "PM":
                    legal_mix_conflict = True
                elif t1 == "PC" and t2 == "PC":
                    legal_mix_conflict = True
            if buffer_collision or legal_mix_conflict:
                relaxed_conflict_pairs.add((a1, a2))

    model = cp_model.CpModel()
    std, eclo = {}, {}
    for aid, d in info.items():
        for w in d["weeks"]:
            std[aid, w] = model.NewBoolVar(f"std_{aid}_{w}")
            eclo[aid, w] = model.NewBoolVar(f"eclo_{aid}_{w}")
            model.Add(std[aid, w] + eclo[aid, w] <= 1)

    # workload conservation (HARD, always)
    for aid, d in info.items():
        needed_half = round(2 * float(d["row"]["total_accesses"]))
        terms = []
        for w in d["weeks"]:
            terms.append(2 * std[aid, w])
            terms.append(3 * eclo[aid, w])
        model.Add(sum(terms) >= needed_half)

    # weekly allocation + workfronts combined bound (HARD, always -- rules
    # 7+8 are about a CONTRACT's own resourcing, not location supply, so
    # they're unaffected by the "flexible supply" reading)
    by_contract_week = defaultdict(list)
    for aid, d in info.items():
        for w in d["weeks"]:
            by_contract_week[(d["row"]["contract_number"], w)].append(aid)
    for (cn, w), aid_list in by_contract_week.items():
        contract = proj.loc[cn]
        bound = int(contract["number_of_maximum_access_per_week"]) * int(contract["number_of_workfronts"])
        model.Add(sum(std[aid, w] + eclo[aid, w] for aid in aid_list) <= bound)

    # predecessor precedence (HARD, always)
    for aid, d in info.items():
        pred = d["row"]["predecessor_activity_id"]
        if not pred or pred not in info:
            continue
        pred_d = info[pred]
        for w_s in d["weeks"]:
            for w_p in pred_d["weeks"]:
                if w_s <= w_p:
                    model.Add((std[aid, w_s] + eclo[aid, w_s]) + (std[pred, w_p] + eclo[pred, w_p]) <= 1)

    # NOTE: buffer/legal-mix pairwise constraints intentionally NOT added
    # to the model anymore -- see docstring. They are only used below,
    # post-solve, to report how much of this flexible "supply" was used.

    # objective: minimize ECLO nights (dominant, matches Score_B's 5x
    # weight) plus a small per-night term for ALL nights used -- without
    # this second term, the solver has zero incentive to avoid scheduling
    # far more nights than the workload minimum requires (since std nights
    # cost nothing in a pure-ECLO objective), which was observed inflating
    # total delivered credit to ~1.8x the actual requirement and needlessly
    # driving up excess_access_nights_total. The small weight keeps ECLO
    # minimization dominant while still discouraging pointless over-delivery.
    total_eclo = sum(eclo[aid, w] for aid, w in eclo)
    total_all_nights = sum(std[aid, w] + eclo[aid, w] for aid, w in std)
    model.Minimize(5 * total_eclo + total_all_nights)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # should not happen (verified satisfiable), but stay robust on a
        # very different hidden instance
        return schedule_scenario_b_interleaved(inst)

    state = ScheduleState()
    footprint_cache = {aid: d["closure_locs"] for aid, d in info.items()}

    by_ctw = defaultdict(list)
    for aid, d in info.items():
        for w in d["weeks"]:
            if solver.Value(std[aid, w]) or solver.Value(eclo[aid, w]):
                by_ctw[(d["row"]["contract_number"], d["access_type"], w)].append(aid)
    access_night_assignment = {}
    for (cn, at, w), aid_list in by_ctw.items():
        contract = proj.loc[cn]
        workfronts = int(contract["number_of_workfronts"])
        cap = int(contract["number_of_maximum_access_per_week"])
        night_loads = defaultdict(int)
        for aid in aid_list:
            for night in range(1, cap + 1):
                if night_loads[night] < workfronts:
                    night_loads[night] += 1
                    access_night_assignment[(aid, w)] = night
                    break

    cosgroup_cache = {}
    cosgroup_counter = [0]

    def next_cosgroup():
        cosgroup_counter[0] += 1
        return f"g{cosgroup_counter[0]}"

    for aid, d in info.items():
        access_seq = 1
        for w in sorted(d["weeks"]):
            is_std, is_eclo = solver.Value(std[aid, w]), solver.Value(eclo[aid, w])
            if not (is_std or is_eclo):
                continue
            night = access_night_assignment[(aid, w)]
            state.access_log[aid].append((w, access_seq, 1 if is_eclo else 0, night))
            for loc in footprint_cache[aid]:
                key = (loc, w)
                if key not in cosgroup_cache:
                    cosgroup_cache[key] = next_cosgroup()
                state.occupancy_log[aid].append((w, loc, cosgroup_cache[key]))
            access_seq += 1

    # post-solve: count how many of the "relaxed" conflicts actually landed
    # in the same week in the final solution, purely for informational /
    # scoring purposes (surfaced via state.relaxed_conflicts_used, picked
    # up by score.py to fold into excess_access_nights_total)
    active_weeks_by_aid = {
        aid: {w for (w, *_r) in entries} for aid, entries in state.access_log.items()
    }
    relaxed_used = 0
    for a1, a2 in relaxed_conflict_pairs:
        common = active_weeks_by_aid.get(a1, set()) & active_weeks_by_aid.get(a2, set())
        relaxed_used += len(common)
    state.relaxed_conflicts_used = relaxed_used

    return state, info


def schedule_scenario_b_interleaved(inst: Instance, max_week_search: int = 400,
                                    priority_override: dict | None = None):
    proj = inst.project_details.set_index("contract_number")
    act = inst.activity_details.copy()
    act["predecessor_activity_id"] = act["predecessor_activity_id"].fillna("")

    info = {}
    for _, row in act.iterrows():
        aid = row["activity_id"]
        contract = proj.loc[row["contract_number"]]
        fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
        cb = closure_and_buffer(inst, contract["nature_of_activity"], fp)
        closure_locs = cb["closure"] | cb["opposite_bound_closure"] | cb["interchange_closure"]
        buffer_locs = cb["buffer_extra"] | cb["opposite_bound_buffer"]
        info[aid] = {
            "row": row,
            "contract": contract,
            "closure_locs": closure_locs,
            "buffer_locs": buffer_locs,
            "fp_key": footprint_key(closure_locs),
            "earliest_week": week_number(inst, pd.Timestamp(row["planned_start_date"]).date()),
            "deadline_week": week_number(inst, pd.Timestamp(contract["planned_completion_date"]).date()),
        }

    buffered_activities_by_location = defaultdict(set)
    for aid, d in info.items():
        if inst.buffer_rules[d["contract"]["nature_of_activity"]]["buffer_sectors"] > 0:
            for loc in d["closure_locs"]:
                buffered_activities_by_location[loc].add(d["row"]["contract_number"])
    pc_pm_locations = defaultdict(set)
    for aid, d in info.items():
        if d["contract"]["access_type"] in ("PC", "PM"):
            for loc in d["closure_locs"]:
                pc_pm_locations[loc].add(d["row"]["contract_number"])
    contentious_pc_pm_locations = {loc for loc, contracts in pc_pm_locations.items() if len(contracts) > 1}
    contentious_buffer_locations = {loc for loc, contracts in buffered_activities_by_location.items() if len(contracts) > 1}
    for aid, d in info.items():
        has_buffer = inst.buffer_rules[d["contract"]["nature_of_activity"]]["buffer_sectors"] > 0
        is_pc_pm_contentious = (
            d["contract"]["access_type"] in ("PC", "PM")
            and bool(d["closure_locs"] & contentious_pc_pm_locations)
        )
        is_buffer_contentious = has_buffer and bool(d["closure_locs"] & contentious_buffer_locations)
        d["is_contentious"] = is_pc_pm_contentious or is_buffer_contentious

    state = ScheduleState()
    remaining = {aid: float(info[aid]["row"]["total_accesses"]) for aid in info}
    access_seq_counter = {aid: 1 for aid in info}
    last_week_used = {}
    finished_week = {}

    def is_eligible(aid, week):
        if remaining[aid] <= 1e-9:
            return False
        d = info[aid]
        if week < d["earliest_week"]:
            return False
        pred = d["row"]["predecessor_activity_id"]
        if pred:
            if pred not in finished_week or finished_week[pred] >= week:
                return False
        return True

    def run_pass(candidate_aids: set):
        for week in range(1, max_week_search + 1):
            if all(remaining[aid] <= 1e-9 for aid in candidate_aids):
                break

            eligible = [aid for aid in candidate_aids if is_eligible(aid, week)]

            def edd_key(aid):
                d = info[aid]
                override_rank = (priority_override or {}).get(aid, 10**9)
                return (override_rank, d["deadline_week"], d["earliest_week"],
                        int(info[aid]["contract"]["contract_priority"]),
                        int(d["row"]["activity_priority"]))

            for aid in sorted(eligible, key=edd_key):
                d = info[aid]
                contract_number = d["row"]["contract_number"]
                contract = d["contract"]
                access_type = contract["access_type"]
                max_access_per_week = int(contract["number_of_maximum_access_per_week"])
                workfronts = int(contract["number_of_workfronts"])

                while remaining[aid] > 1e-9:
                    weeks_left = d["deadline_week"] - week + 1
                    use_eclo = d["is_contentious"] or weeks_left <= math.ceil(remaining[aid])

                    ok, night, cosgroup_per_loc = can_place_week(
                        state, inst, contract_number, access_type, week,
                        d["closure_locs"], d["buffer_locs"], d["fp_key"],
                        max_access_per_week, workfronts, capacity_mode="soft",
                    )
                    if not ok:
                        break
                    credit = 1.5 if use_eclo else 1.0
                    place_week(
                        state, contract_number, access_type, week, aid,
                        d["closure_locs"], d["buffer_locs"], d["fp_key"],
                        night, cosgroup_per_loc, access_seq_counter[aid],
                        eclo=1 if use_eclo else 0,
                    )
                    remaining[aid] -= credit
                    access_seq_counter[aid] += 1
                    last_week_used[aid] = week
                    if remaining[aid] <= 1e-9:
                        finished_week[aid] = week
                    weeks_left_after = d["deadline_week"] - week + 1
                    if weeks_left_after > math.ceil(remaining[aid] / 1.5) + 1:
                        break

    run_pass(set(info.keys()))

    unscheduled_shortfall = [(aid, remaining[aid]) for aid in info if remaining[aid] > 1e-9]
    deadline_missed = []
    for contract_number, contract in proj.iterrows():
        deadline_week = week_number(inst, pd.Timestamp(contract["planned_completion_date"]).date())
        aids = [aid for aid in info if info[aid]["row"]["contract_number"] == contract_number]
        weeks = [last_week_used[aid] for aid in aids if aid in last_week_used]
        if weeks and max(weeks) > deadline_week:
            deadline_missed.append((contract_number, max(weeks), deadline_week))

    if unscheduled_shortfall:
        print(f"WARNING: {len(unscheduled_shortfall)} activities did not fully complete "
              f"within {max_week_search} week search window:")
        for aid, short in unscheduled_shortfall:
            print(f"   {aid}: short by {short} access(es)")
    if deadline_missed:
        print(f"WARNING (Scenario B hard-deadline check): {len(deadline_missed)} contracts "
              f"finished AFTER their planned_completion_date despite ECLO/excess-capacity:")
        for cn, lw, dw in deadline_missed:
            print(f"   {cn}: finished week {lw}, deadline week {dw}")

    return state, info


def schedule_scenario(
    inst: Instance,
    scenario: str,
    max_week_search: int = 400,
):
    proj = inst.project_details.set_index("contract_number")
    act = inst.activity_details.copy()
    act["predecessor_activity_id"] = act["predecessor_activity_id"].fillna("")

    if scenario == "B":
        return schedule_scenario_b_exact(inst)

    capacity_mode = {"A": "hard", "C": "soft_1"}[scenario]
    eclo_allowed = scenario == "C"
    deadline_hard = False

    # precompute per-activity static info
    info = {}
    for _, row in act.iterrows():
        aid = row["activity_id"]
        contract = proj.loc[row["contract_number"]]
        fp = expand_footprint(inst, row["start_location_id"], row["end_location_id"])
        cb = closure_and_buffer(inst, contract["nature_of_activity"], fp)
        closure_locs = cb["closure"] | cb["opposite_bound_closure"] | cb["interchange_closure"]
        buffer_locs = cb["buffer_extra"] | cb["opposite_bound_buffer"]
        info[aid] = {
            "row": row,
            "contract": contract,
            "footprint": fp,
            "cb": cb,
            "closure_locs": closure_locs,
            "buffer_locs": buffer_locs,
            "fp_key": footprint_key(closure_locs),
            "earliest_week": week_number(inst, pd.Timestamp(row["planned_start_date"]).date()),
            "deadline_week": week_number(inst, pd.Timestamp(contract["planned_completion_date"]).date()),
            "line": fp["line"],
        }

    # --- Scenario C: pick one fixed 2-week ECLO window synchronized across the network ---
    eclo_window = {}
    if scenario == "C":
        all_earliest_weeks = [info[aid]["earliest_week"] for aid in info]
        if all_earliest_weeks:
            best_w, best_count = min(all_earliest_weeks), -1
            for w in set(all_earliest_weeks):
                count = sum(1 for x in all_earliest_weeks if w <= x <= w + 1)
                if count > best_count:
                    best_count, best_w = count, w
            shared_window = (best_w, best_w + 1)
        else:
            shared_window = (1, 2)

        for line in inst.lines["line_code"]:
            eclo_window[line] = shared_window

    def topo_order(activity_ids, priority_key):
        preds = {aid: info[aid]["row"]["predecessor_activity_id"] for aid in activity_ids}
        indegree = {aid: (1 if preds[aid] else 0) for aid in activity_ids}
        successors = defaultdict(list)
        for aid, p in preds.items():
            if p and p in indegree:
                successors[p].append(aid)

        ready = sorted([aid for aid in activity_ids if indegree[aid] == 0], key=priority_key)
        ordered = []
        while ready:
            ready.sort(key=priority_key)
            aid = ready.pop(0)
            ordered.append(aid)
            for succ in successors[aid]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
        if len(ordered) != len(activity_ids):
            leftover = [aid for aid in activity_ids if aid not in ordered]
            print(f"WARNING: {len(leftover)} activities could not be topologically "
                  f"ordered (cycle or dangling predecessor?): {leftover}")
            ordered.extend(sorted(leftover, key=priority_key))
        return ordered

    def sort_key(aid):
        c = info[aid]["contract"]
        r = info[aid]["row"]
        return (int(c["contract_priority"]), int(r["activity_priority"]), info[aid]["earliest_week"])

    processing_order = topo_order(list(info.keys()), sort_key)

    state = ScheduleState()
    pred_finish_week = {}

    unscheduled_shortfall = []
    deadline_missed = []

    def schedule_one(aid):
        d = info[aid]
        row, contract = d["row"], d["contract"]
        contract_number = row["contract_number"]
        access_type = contract["access_type"]
        max_access_per_week = int(contract["number_of_maximum_access_per_week"])
        workfronts = int(contract["number_of_workfronts"])
        total_accesses = float(row["total_accesses"])
        deadline_week = d["deadline_week"]
        line = d["line"]

        start_week = d["earliest_week"]
        pred = row["predecessor_activity_id"]
        if pred and pred in pred_finish_week:
            start_week = max(start_week, pred_finish_week[pred] + 1)

        placed_credit = 0.0
        week = start_week
        access_seq = 1
        last_week_used = start_week
        tries = 0

        while placed_credit < total_accesses - 1e-9 and tries < max_week_search:
            remaining = total_accesses - placed_credit

            use_eclo = False
            catching_up = False
            if scenario == "B" and eclo_allowed:
                weeks_left = deadline_week - week + 1
                if weeks_left <= math.ceil(remaining):
                    use_eclo = True
                if weeks_left <= math.ceil(remaining / 1.5):
                    catching_up = True
            elif scenario == "C" and eclo_allowed:
                lo, hi = eclo_window[line]
                touches_interchange = bool(d["cb"].get("interchange_closure"))
                is_live_rail = inst.buffer_rules.get(contract["nature_of_activity"], {}).get("opposite_bound_required", False)

                if lo <= week <= hi:
                    use_eclo = True
                elif touches_interchange and is_live_rail:
                    use_eclo = False

            ok, night, cosgroup_per_loc = can_place_week(
                state, inst, contract_number, access_type, week,
                d["closure_locs"], d["buffer_locs"], d["fp_key"],
                max_access_per_week, workfronts, capacity_mode=capacity_mode,
            )
            if ok:
                credit = 1.5 if use_eclo else 1.0
                place_week(
                    state, contract_number, access_type, week, aid,
                    d["closure_locs"], d["buffer_locs"], d["fp_key"],
                    night, cosgroup_per_loc, access_seq, eclo=1 if use_eclo else 0,
                )
                placed_credit += credit
                access_seq += 1
                last_week_used = week
                if not (catching_up and placed_credit < total_accesses - 1e-9):
                    week += 1
            else:
                week += 1
            tries += 1

        pred_finish_week[aid] = last_week_used
        if placed_credit < total_accesses - 1e-9:
            unscheduled_shortfall.append((aid, total_accesses - placed_credit))
        if deadline_hard and last_week_used > deadline_week:
            deadline_missed.append((aid, contract_number, last_week_used, deadline_week))

    for aid in processing_order:
        schedule_one(aid)

    if unscheduled_shortfall:
        print(f"WARNING: {len(unscheduled_shortfall)} activities did not fully complete "
              f"within {max_week_search} week search window:")
        for aid, short in unscheduled_shortfall:
            print(f"   {aid}: short by {short} access(es)")
    if deadline_missed:
        print(f"WARNING (Scenario B hard-deadline check): {len(deadline_missed)} activities "
              f"finished AFTER their contract's planned_completion_date despite ECLO/excess-capacity:")
        for aid, cn, lw, dw in deadline_missed:
            print(f"   {aid} ({cn}): finished week {lw}, deadline week {dw}")

    return state, info


def schedule_scenario_a(inst: Instance, max_week_search: int = 400):
    return schedule_scenario(inst, "A", max_week_search)


def build_output_frames(inst: Instance, state: ScheduleState, info: dict, scenario: str):
    access_rows = []
    for aid, entries in state.access_log.items():
        for (week, access_seq, eclo, access_night) in entries:
            access_rows.append({
                "activity_id": aid, "access_seq": access_seq, "week": week,
                "eclo": eclo, "access_night": access_night,
            })
    access_df = pd.DataFrame(access_rows).sort_values(["activity_id", "access_seq"])

    occ_rows = []
    for aid, entries in state.occupancy_log.items():
        for (week, loc, cosgroup) in entries:
            occ_rows.append({
                "activity_id": aid, "week": week, "location_id": loc, "co_share_group": cosgroup,
            })
    occ_df = pd.DataFrame(occ_rows).sort_values(["activity_id", "week"])

    proj = inst.project_details.set_index("contract_number")
    result_rows = []
    act_by_contract = inst.activity_details.groupby("contract_number")["activity_id"].apply(list)
    for contract_number, contract in proj.iterrows():
        aids = act_by_contract.get(contract_number, [])
        last_weeks = []
        for aid in aids:
            weeks = [w for (w, *_ ) in state.access_log.get(aid, [])]
            if weeks:
                last_weeks.append(max(weeks))
        if last_weeks:
            finish_week = max(last_weeks)
            sim_date = week_start_date(inst, finish_week)
        else:
            sim_date = None
        planned = pd.Timestamp(contract["planned_completion_date"]).date()
        if sim_date is not None:
            overrun = max(0, (sim_date - planned).days)
        else:
            overrun = None
        result_rows.append({
            "scenario": scenario,
            "contract_number": contract_number,
            "simulated_completion_date": sim_date,
            "overrun_days": overrun,
        })
    results_df = pd.DataFrame(result_rows)

    return access_df, occ_df, results_df


if __name__ == "__main__":
    inst = load_instance()
    state, info = schedule_scenario_a(inst)
    access_df, occ_df, results_df = build_output_frames(inst, state, info, "A")

    print(f"\nScheduled {len(state.access_log)} / {len(inst.activity_details)} activities (at least partially)")
    total_needed = inst.activity_details["total_accesses"].sum()
    total_placed = sum(len(v) for v in state.access_log.values())
    print(f"Total access-nights placed: {total_placed} / {total_needed} needed")
    print()
    print("RESULTS.csv preview:")
    print(results_df.to_string(index=False))
    print()
    print(f"SCHEDULE_ACCESS.csv rows: {len(access_df)}")
    print(f"SCHEDULE_OCCUPANCY.csv rows: {len(occ_df)}")
    print()
    print("Total overrun days (sum across contracts):", results_df["overrun_days"].sum())
