"""
Natural-language Q&A over a generated schedule, powered by Google's Gemini
API. This is the "Natural Language Querying" bonus feature from the problem
statement's bonus scope (root-cause of a move, downstream delay risk,
capacity/co-sharing checks, milestone risk, handover briefs).

Uses the current unified Google GenAI SDK (`pip install google-genai`,
NOT the deprecated `google-generativeai`). Defaults to the Gemini
Developer API (a plain API key from https://aistudio.google.com/apikey),
since that's the fastest path to a working demo. If you'd rather route
through the GCP project the hackathon issued you (so the judges see it
hitting your actual Google Cloud project), swap the client construction
in get_client() for:

    genai.Client(vertexai=True, project="<your-qwiklabs-project-id>", location="us-central1")

-- everything else (model calls, context building) stays identical, since
the google-genai SDK exposes the same interface for both backends.
"""
from __future__ import annotations

import re

import pandas as pd

from data_model import Instance, humanize_location

SYSTEM_PROMPT = """You are a scheduling assistant for a railway works controller \
using a track-access possession scheduler. You answer questions about a \
SPECIFIC generated schedule, using ONLY the facts given to you in the \
CONTEXT section below -- never invent contract numbers, activity IDs, \
dates, or locations that aren't present in the context.

Rules:
- Be concise and direct. A works controller at 2am wants the answer, not a lecture.
- If a USER GOAL is given below, tailor tone and depth to it -- e.g. someone
  "checking tonight's risk" wants a fast verdict, someone "preparing a report
  for a manager" wants slightly more framing/context. Never invent a goal if
  none is given; just answer plainly.
- When referring to a location, lead with its plain-English description (given in \
  brackets next to each location_id in the context) rather than the raw code -- \
  e.g. say "the Beta Line platform at H02 (Eastbound)" not "PLAT:BET:H02:EB". \
  You may mention the raw ID once for traceability if useful, but never lead with it.
- Always cite the specific contract_number / activity_id / week involved.
- If the context doesn't contain enough information to answer, say so plainly \
  rather than guessing.
- "Week N" means week N of the scheduling horizon (see the horizon_start date \
  in the context to convert to a calendar date if asked).
- If asked WHY something happened (a delay, a capacity issue), reason from the \
  hotspot and violation data given, not from general knowledge of railways.
"""


def build_static_context(inst: Instance, access_df: pd.DataFrame, occ_df: pd.DataFrame,
                          results_df: pd.DataFrame, report: dict, scenario: str) -> str:
    """Always-included summary: small enough to stay cheap, rich enough to
    answer most 'why is X delayed / what's the status of Y' questions
    without needing entity-specific drill-down."""
    lines = [f"=== SCENARIO {scenario} ===",
             f"Horizon start: {inst.horizon_start}, {inst.horizon_weeks} weeks.",
             f"Feasible: {report['feasible']}. Hard violations: {len(report['hard_violations'])}.",
             ""]

    if report["hard_violations"]:
        lines.append("HARD VIOLATIONS:")
        for v in report["hard_violations"][:20]:
            lines.append(f"  [{v['rule']}] {v['detail']}")
        lines.append("")

    ss = report["soft_scores"]
    lines.append(
        f"Soft scores: overrun_days_total={ss['overrun_days_total']}, "
        f"contracts_overrunning={ss['contracts_overrunning']}, "
        f"excess_access_nights_total={ss['excess_access_nights_total']}, "
        f"eclo_nights_total={ss['eclo_nights_total']}, "
        f"priority_overrun={ss['priority_overrun']}."
    )
    lines.append("")

    lines.append("CONTRACT COMPLETION SUMMARY (contract_number, priority, planned date, "
                  "simulated date, overrun_days):")
    proj = inst.project_details.set_index("contract_number")
    for _, row in results_df.iterrows():
        cn = row["contract_number"]
        priority = proj.loc[cn, "contract_priority"] if cn in proj.index else "?"
        lines.append(f"  {cn} (priority {priority}): planned={proj.loc[cn, 'planned_completion_date']}, "
                      f"simulated={row['simulated_completion_date']}, overrun_days={row['overrun_days']}")
    lines.append("")

    lines.append("TOP 10 CAPACITY HOTSPOTS (location_id: distinct activities using it):")
    hotspot = occ_df.groupby("location_id")["activity_id"].nunique().sort_values(ascending=False).head(10)
    for loc, n in hotspot.items():
        cap = inst.supply.get(loc, "?")
        lines.append(f"  {loc} [{humanize_location(inst, loc)}]: {n} activities (nominal supply_capacity={cap})")

    return "\n".join(lines)


def build_entity_context(inst: Instance, access_df: pd.DataFrame, occ_df: pd.DataFrame,
                          question: str) -> str:
    """Regex-pull any contract/activity/location IDs mentioned in the
    question and attach their full detail, so answers about a SPECIFIC
    entity are grounded in its actual records rather than the summary
    alone.

    Also auto-expands: any location a referenced contract/activity actually
    uses, IF that location is one of the global top-10 congestion hotspots,
    gets its own per-week occupant listing pulled in automatically -- this
    is what lets "why is C001 overrunning" get answered by cross-referencing
    which of C001's locations were contested and by whom, without the user
    needing to already know and name the specific bottleneck location."""
    lines = []
    act = inst.activity_details.set_index("activity_id")
    proj = inst.project_details.set_index("contract_number")

    contract_ids = sorted(set(re.findall(r"\bC\d{3,}\b", question)))
    activity_ids = sorted(set(re.findall(r"\bA\d{3,}\b", question)))
    location_ids = set(re.findall(r"\b(?:SEC|PLAT):[A-Za-z0-9_:]+\b", question))

    hotspot_locations = set(
        occ_df.groupby("location_id")["activity_id"].nunique().sort_values(ascending=False).head(10).index
    )
    auto_expand_locations = set()

    for cn in contract_ids:
        if cn not in proj.index:
            continue
        c = proj.loc[cn]
        lines.append(f"--- CONTRACT {cn} ---")
        lines.append(f"  {c.to_dict()}")
        its_activities = act[act["contract_number"] == cn]
        for aid, a in its_activities.iterrows():
            weeks = sorted(access_df[access_df["activity_id"] == aid]["week"].tolist())
            locs = occ_df[occ_df["activity_id"] == aid]["location_id"].unique().tolist()
            lines.append(f"  activity {aid}: {a['start_location_id']} -> {a['end_location_id']}, "
                         f"total_accesses={a['total_accesses']}, scheduled weeks={weeks}, "
                         f"locations used={[f'{l} [{humanize_location(inst, l)}]' for l in locs]}")
            auto_expand_locations.update(loc for loc in locs if loc in hotspot_locations)

    for aid in activity_ids:
        if aid not in act.index:
            continue
        a = act.loc[aid]
        weeks = sorted(access_df[access_df["activity_id"] == aid]["week"].tolist())
        locs = occ_df[occ_df["activity_id"] == aid]["location_id"].unique().tolist()
        lines.append(f"--- ACTIVITY {aid} ---")
        lines.append(f"  contract={a['contract_number']}, {a['start_location_id']} -> {a['end_location_id']}, "
                     f"total_accesses={a['total_accesses']}, predecessor={a['predecessor_activity_id']}")
        lines.append(f"  scheduled weeks: {weeks}")
        lines.append(f"  locations booked: {locs}")
        auto_expand_locations.update(loc for loc in locs if loc in hotspot_locations)

    all_locations_to_expand = location_ids | auto_expand_locations
    for loc in all_locations_to_expand:
        sub = occ_df[occ_df["location_id"] == loc]
        cap = inst.supply.get(loc, "unknown")
        auto_tag = " [auto-included: this is one of the top-10 contested locations " \
                   "and is used by an activity/contract in your question]" if loc in auto_expand_locations else ""
        lines.append(f"--- LOCATION {loc} [{humanize_location(inst, loc)}] (nominal capacity {cap}){auto_tag} ---")
        by_week = sub.groupby("week")["activity_id"].apply(list)
        for week, aids in by_week.items():
            lines.append(f"  week {week}: {aids}")

    return "\n".join(lines) if lines else ""


def get_client(api_key: str | None = None, use_vertex: bool = False,
                project: str | None = None, location: str = "us-central1"):
    from google import genai
    if use_vertex:
        # Vertex AI uses Google Cloud IAM auth (Application Default
        # Credentials), not an API key. On Cloud Run this is picked up
        # AUTOMATICALLY from the service's own identity -- no key, no
        # secret, nothing to configure in the app itself. Locally, it
        # needs `gcloud auth application-default login` first.
        return genai.Client(vertexai=True, project=project, location=location)
    if api_key:
        return genai.Client(api_key=api_key)
    # neither explicit api_key nor use_vertex given: fall back to whatever
    # GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
    # environment variables are set -- lets the deployment environment pick
    # the backend with zero code changes
    return genai.Client()


def ask_gemini(question: str, context: str, api_key: str = "", model: str = "gemini-2.5-flash",
                use_vertex: bool = False, project: str | None = None,
                location: str = "us-central1", user_intent: str = "") -> str:
    from google.genai import types

    client = get_client(api_key=api_key, use_vertex=use_vertex, project=project, location=location)
    intent_block = f"USER GOAL: {user_intent}\n\n" if user_intent.strip() else ""
    prompt = f"{intent_block}CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=1024,
            # This is a quick factual lookup over data already handed to the
            # model, not a task needing deep reasoning -- gemini-2.5-flash
            # has "thinking" ON by default, and thinking tokens are drawn
            # from the SAME budget as max_output_tokens, which is exactly
            # what was truncating answers before this was added.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text
