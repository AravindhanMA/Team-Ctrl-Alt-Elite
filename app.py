import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_model import load_instance, week_number, humanize_location
from scheduler import schedule_scenario, build_output_frames
from validator import validate
from score import compute_soft_scores

st.set_page_config(page_title="Nebula X PS1 - Track Access Scheduler", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #10151f;
    --bg-card: #131926;
    --border-color: rgba(148, 163, 184, 0.14);
    --accent: #22d3ee;
    --accent-glow: rgba(34, 211, 238, 0.16);
    --accent-secondary: #818cf8;
    --text-primary: #e7ebf3;
    --text-secondary: #8b96a8;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
}

.stApp {
    background: radial-gradient(ellipse 1200px 800px at 50% -10%, #131b2e 0%, #0a0e17 55%);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

h1 {
    background: linear-gradient(120deg, #ffffff 10%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding-bottom: 4px;
}

.kicker {
    color: var(--accent);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px 10px 18px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent);
    box-shadow: 0 0 28px var(--accent-glow);
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--border-color);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    transition: all 0.18s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, var(--accent) 0%, var(--accent-secondary) 100%);
    border: none;
    color: #061018;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px var(--accent-glow);
}

[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px;
}

[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--border-color);
}
[aria-selected="true"] {
    color: var(--accent) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
}

hr {
    border-color: var(--border-color) !important;
}

.status-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-feasible {
    background: rgba(52, 211, 153, 0.12);
    color: var(--success);
    border: 1px solid rgba(52, 211, 153, 0.32);
}
.status-warning {
    background: rgba(251, 191, 36, 0.12);
    color: var(--warning);
    border: 1px solid rgba(251, 191, 36, 0.32);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

REQUIRED_FILES = [
    "01_LINES.csv", "02_STATIONS.csv", "03_SECTORS.csv", "04_LOCATION_SUPPLY.csv",
    "05_BUFFER_LOCATION.csv", "06_PARAMETERS.csv", "07_PROJECT_DETAILS.csv",
    "08_ACTIVITY_DETAILS.csv",
]

st.markdown('<p class="kicker">Track Access Intelligence</p>', unsafe_allow_html=True)
st.title("Railway Track Access Scheduler")
st.caption("Nebula X Hackathon 2026 — Problem Statement 1")

with st.sidebar:
    st.header("1. Load instance")
    uploaded = st.file_uploader(
        "Upload all 8 instance CSVs (or a .zip containing them)",
        type=["csv", "zip"],
        accept_multiple_files=True,
    )
    use_sample = st.checkbox("Use bundled sample instance instead", value=not uploaded)

    st.header("2. Choose scenarios")
    run_a = st.checkbox("Scenario A (strict supply)", value=True,
                         help="Track capacity is fixed and never exceeded. Schedule pressure is "
                              "absorbed entirely as delay — no early closures, no extra capacity.")
    run_b = st.checkbox("Scenario B (strict schedule)", value=True,
                         help="Planned completion dates are fixed and must be hit. Capacity can be "
                              "exceeded and early closures used freely to make that happen.")
    run_c = st.checkbox("Scenario C (balanced)", value=True,
                         help="A realistic middle ground — a little schedule slip AND a little extra "
                              "capacity/early-closure, balancing both rather than maxing out either.")

    run_clicked = st.button("▶ Run scheduler", type="primary", width='stretch')

    st.header("3. AI Q&A")
    _env_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if _env_project:
        # Deployed on Cloud Run with Vertex AI configured at deploy time
        # (--set-env-vars) -- just use it silently. A works controller has
        # no reason to see or touch a GCP project ID / region.
        st.session_state["ai_backend"] = "vertex"
        st.session_state["vertex_project"] = _env_project
        st.session_state["vertex_location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        st.caption("AI Assistant Ready — Powered by Google Cloud Vertex AI")
    else:
        # Local/dev fallback only (no env var set): surface manual config,
        # since there's no deployment environment to infer it from.
        ai_backend = st.radio(
            "How should Q&A authenticate to Gemini?",
            ["Gemini API key", "Vertex AI (needs `gcloud auth application-default login`)"],
            index=0,
            help="On the deployed app this is configured automatically and this "
                 "choice won't appear — it's only shown here for local testing.",
        )
        st.session_state["ai_backend"] = "vertex" if ai_backend.startswith("Vertex") else "api_key"
        if st.session_state["ai_backend"] == "vertex":
            st.session_state["vertex_project"] = st.text_input("GCP Project ID", value="")
            st.session_state["vertex_location"] = st.text_input("Region", value="us-central1")
        else:
            st.session_state["gemini_api_key"] = st.text_input(
                "Gemini API key", type="password",
                help="Get a free key at https://aistudio.google.com/apikey.",
                value=st.session_state.get("gemini_api_key", ""),
            )

    with st.form(key="intent_form", clear_on_submit=False):
        intent_input = st.text_input(
            "What are you working on right now? (optional)",
            placeholder="e.g. checking tonight's risk, prepping a report for my manager...",
            value=st.session_state.get("user_intent", ""),
            help="Tells the AI assistant what you're trying to do, so its answers "
                 "are framed for that — a quick risk check gets a fast verdict, "
                 "a report gets a bit more context. This doesn't answer questions "
                 "itself; it shapes every AI answer elsewhere in the app.",
        )
        intent_submitted = st.form_submit_button("Save")
    if intent_submitted:
        st.session_state["user_intent"] = intent_input
        if intent_input.strip():
            st.success(f"Got it — the AI assistant will frame its answers around: \"{intent_input}\"")
        else:
            st.caption("Cleared — the AI assistant will answer plainly.")


def materialize_instance_dir(uploaded_files) -> str:
    tmpdir = tempfile.mkdtemp()
    for f in uploaded_files:
        if f.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                z.extractall(tmpdir)
        else:
            (Path(tmpdir) / f.name).write_bytes(f.getbuffer())
    missing = [name for name in REQUIRED_FILES if not (Path(tmpdir) / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    return tmpdir


@st.cache_data(show_spinner=False)
def run_scenario_cached(data_dir: str, scenario: str):
    inst = load_instance(data_dir)
    state, info = schedule_scenario(inst, scenario)
    access_df, occ_df, results_df = build_output_frames(inst, state, info, scenario)
    violations = validate(inst, access_df, occ_df, scenario)
    report = compute_soft_scores(inst, access_df, occ_df, results_df, scenario)
    report["feasible"] = len(violations) == 0
    report["hard_violations"] = violations
    return access_df, occ_df, results_df, report


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_zip(access_df, occ_df, results_df, report) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SCHEDULE_ACCESS.csv", access_df.to_csv(index=False))
        z.writestr("SCHEDULE_OCCUPANCY.csv", occ_df.to_csv(index=False))
        z.writestr("RESULTS.csv", results_df.to_csv(index=False))
        z.writestr("validation_report.json", json.dumps(report, indent=2, default=str))
    return buf.getvalue()


def render_comparison(scenarios: list, data_dir: str):
    """The core trade-off story of PS1, in one view: A absorbs pressure as
    pure delay, B eliminates delay by spending unlimited capacity + ECLO,
    C is the balanced middle. Side by side, not three separate tabs."""
    st.header("Scenario Comparison")
    rows = []
    for s in scenarios:
        access_df, occ_df, results_df, report = run_scenario_cached(data_dir, s)
        ss = report["soft_scores"]
        rows.append({
            "Scenario": s,
            "Feasible": "Yes" if report["feasible"] else "No",
            "Hard violations": len(report["hard_violations"]),
            "Overrun (days)": ss["overrun_days_total"],
            "Excess access-nights": ss["excess_access_nights_total"],
            "ECLO nights": ss["eclo_nights_total"],
            "Objective score": report["objective_score"],
        })
    comp_df = pd.DataFrame(rows).set_index("Scenario")
    st.dataframe(comp_df, width='stretch')

    st.caption(
        "Scenario A absorbs all schedule pressure as pure delay (overrun only, capacity never "
        "flexed). Scenario B removes delay entirely by spending unlimited extra capacity + ECLO "
        "nights. Scenario C is the balanced middle ground — a little of both. Lower is better on "
        "every column except Feasible."
    )
    st.bar_chart(comp_df[["Overrun (days)", "Excess access-nights", "ECLO nights"]])
    render_inline_ai_note(
        "comparison",
        "SCENARIO COMPARISON TABLE:\n" + comp_df.to_string(),
        placeholder="e.g. Which scenario should I actually pick, and why?",
    )
    st.divider()


def render_timeline(scenario: str, inst, access_df: pd.DataFrame, occ_df: pd.DataFrame,
                     results_df: pd.DataFrame, report: dict, data_dir: str):
    st.subheader("Schedule Timeline")
    act = inst.activity_details.set_index("activity_id")
    proj = inst.project_details.set_index("contract_number")

    merged = access_df.merge(act[["contract_number"]], left_on="activity_id", right_index=True)
    if merged.empty:
        st.info("No accesses scheduled — nothing to show on the timeline.")
        return
    spans = merged.groupby("contract_number")["week"].agg(["min", "max"]).reset_index()
    spans.columns = ["contract_number", "first_week", "last_week"]
    spans = spans.merge(results_df[["contract_number", "overrun_days"]], on="contract_number", how="left")
    spans["overrun_days"] = spans["overrun_days"].fillna(0)
    spans["deadline_week"] = spans["contract_number"].apply(
        lambda cn: week_number(inst, pd.Timestamp(proj.loc[cn, "planned_completion_date"]).date())
    )
    spans = spans.sort_values("first_week")

    def bar_color(d):
        if d <= 0:
            return "#34d399"
        elif d < 30:
            return "#fbbf24"
        return "#f87171"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=spans["contract_number"], x=spans["last_week"] - spans["first_week"] + 1,
        base=spans["first_week"], orientation="h",
        marker_color=[bar_color(d) for d in spans["overrun_days"]],
        name="Scheduled span",
        hovertext=[f"{cn}: weeks {fw}-{lw}, overrun {od:.0f}d"
                   for cn, fw, lw, od in zip(spans.contract_number, spans.first_week,
                                              spans.last_week, spans.overrun_days)],
        hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=spans["deadline_week"], y=spans["contract_number"], mode="markers",
        marker=dict(symbol="line-ns-open", size=18, color="#e7ebf3", line=dict(width=3)),
        name="Planned deadline",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e7ebf3"),
        title=f"Scenario {scenario}: contract activity span vs. planned deadline (black tick) "
              "— green=on time, orange=<30d late, red=30d+ late",
        xaxis_title="Week", yaxis_title="Contract",
        height=max(300, 32 * len(spans)), margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, width='stretch')
    render_inline_ai_note(
        f"gantt_{scenario}",
        f"SCENARIO {scenario} CONTRACT TIMELINE (deadline_week vs actual span):\n"
        + spans[["contract_number", "first_week", "last_week", "deadline_week", "overrun_days"]].to_string(index=False),
        placeholder="e.g. Why does this contract span so many weeks?",
        scenario=scenario, data_dir=data_dir, access_df=access_df, occ_df=occ_df,
        results_df=results_df, report=report,
    )

    st.markdown("**Drill into one contract's activities:**")
    selected = st.selectbox("Contract", spans["contract_number"].tolist(), key=f"drill_{scenario}")
    its_activities = act[act["contract_number"] == selected].index.tolist()
    sub = access_df[access_df["activity_id"].isin(its_activities)]
    if sub.empty:
        st.info("No accesses scheduled for this contract.")
        return
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sub["week"], y=sub["activity_id"], mode="markers",
        marker=dict(size=11, color=sub["eclo"].map({0: "#22d3ee", 1: "#818cf8"})),
        hovertext=[f"week {w}, {'ECLO' if e else 'standard'} night" for w, e in zip(sub.week, sub.eclo)],
        hoverinfo="text",
    ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e7ebf3"),
        title=f"{selected}: each dot is one scheduled access night (violet = ECLO)",
        xaxis_title="Week", yaxis_title="Activity",
        height=max(220, 32 * len(its_activities)), margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig2, width='stretch')
    render_inline_ai_note(
        f"drill_{scenario}_{selected}",
        f"ACTIVITY-LEVEL DETAIL for {selected} (scenario {scenario}):\n"
        + sub[["activity_id", "week", "eclo"]].to_string(index=False),
        placeholder=f"e.g. Why is {selected} using ECLO on these nights?",
        scenario=scenario, data_dir=data_dir, access_df=access_df, occ_df=occ_df,
        results_df=results_df, report=report, extra_entity_hint=selected,
    )


def render_briefing(inst, results_df: pd.DataFrame, occ_df: pd.DataFrame, report: dict, scenario: str):
    """Plain-language, template-based (no AI call needed -- fast and
    reliable) summary for a tired 2am works controller: what's the status,
    right now, in one read, before any chart or table."""
    st.subheader("Schedule Briefing")
    disp = results_df.copy()
    disp["overrun_days"] = disp["overrun_days"].fillna(0)
    n_total = len(disp)
    n_atrisk = int((disp["overrun_days"] > 0).sum())
    n_ontrack = n_total - n_atrisk

    if report["feasible"]:
        st.markdown('<span class="status-badge status-feasible">Feasible</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">Attention Required</span>', unsafe_allow_html=True)

    lines = []
    if report["feasible"]:
        lines.append("**This schedule is feasible** — every hard safety and allocation rule is satisfied.")
    else:
        lines.append(f"**Not yet fully feasible** — {len(report['hard_violations'])} hard rule "
                      f"violation(s) remain (see the violations table below).")

    if n_atrisk == 0:
        lines.append(f"All **{n_total} contracts** are on track to finish on or before their planned date.")
    else:
        worst = disp.loc[disp["overrun_days"].idxmax()]
        lines.append(
            f"**{n_ontrack} of {n_total} contracts** are on track. "
            f"**{n_atrisk} are running late** — worst is **{worst['contract_number']}**, "
            f"**{worst['overrun_days']:.0f} days** behind schedule."
        )

    hotspot = occ_df.groupby("location_id")["activity_id"].nunique().sort_values(ascending=False)
    if len(hotspot) > 0:
        top_loc, top_n = hotspot.index[0], hotspot.iloc[0]
        cap = inst.supply.get(top_loc, "?")
        lines.append(
            f"Busiest bottleneck: **{humanize_location(inst, top_loc)}** — "
            f"{top_n} different activities compete for it against a capacity of just {cap}."
        )

    ss = report["soft_scores"]
    extras = []
    if ss["eclo_nights_total"] > 0:
        extras.append(f"{ss['eclo_nights_total']} early-closure (ECLO — extended work hours per "
                       f"night) nights used to help meet deadlines")
    if ss["excess_access_nights_total"] > 0:
        extras.append(f"{ss['excess_access_nights_total']} access-nights exceeded nominal capacity to relieve congestion")
    if extras:
        lines.append(" · ".join(extras) + ".")

    st.info("\n\n".join(lines))


def _ai_backend_status():
    """Shared readiness check used by every AI touchpoint in the app (the
    main Ask-the-Scheduler panel AND every per-chart inline note) so they
    all behave identically and there's one place to fix if the backend
    logic ever changes."""
    backend = st.session_state.get("ai_backend", "vertex")
    if backend == "vertex":
        vertex_project = st.session_state.get("vertex_project", "")
        vertex_location = st.session_state.get("vertex_location", "us-central1")
        return backend, bool(vertex_project), {"project": vertex_project, "location": vertex_location}
    else:
        gemini_key = st.session_state.get("gemini_api_key", "")
        return backend, bool(gemini_key), {"api_key": gemini_key}


def _call_gemini(question: str, context: str) -> str:
    from qa import ask_gemini
    backend, ready, creds = _ai_backend_status()
    user_intent = st.session_state.get("user_intent", "")
    if backend == "vertex":
        return ask_gemini(question, context, use_vertex=True, project=creds["project"],
                           location=creds["location"], user_intent=user_intent)
    return ask_gemini(question, context, api_key=creds["api_key"], user_intent=user_intent)


def render_inline_ai_note(key_suffix: str, chart_context: str, placeholder: str = "Ask about this chart...",
                           scenario: str | None = None, data_dir: str | None = None,
                           access_df=None, occ_df=None, results_df=None, report: dict | None = None,
                           extra_entity_hint: str = ""):
    """A compact, per-visualization AI note -- reuses the same Vertex AI /
    Gemini backend as the main Ask-the-Scheduler panel, but scoped to just
    the data behind THIS specific chart, so a planner can ask a quick
    follow-up right where they're looking rather than hunting for the main
    Q&A section. Collapsed by default (an expander) so it stays out of the
    way until someone actually wants it -- "lowkey" interactive, not a
    wall of chat boxes.

    When scenario/data_dir/access_df/occ_df/results_df/report are all
    supplied, this ALSO pulls in the same static + entity-expansion context
    the main Q&A panel uses (feasibility, violations, hotspots, and a full
    location/activity breakdown of anything the question or extra_entity_hint
    names) -- without this, "why is this contract delayed" style questions
    could only see the thin chart-specific numbers (e.g. just
    first_week/last_week/deadline_week) and had nothing to reason a root
    cause FROM, which is exactly what was producing "context doesn't
    contain enough information" non-answers. extra_entity_hint lets a
    caller that already knows the specific subject of its chart (e.g. the
    contract selected in a drill-down) force entity-expansion for it even
    if the user's own question doesn't happen to name it explicitly.
    """
    _, ready, _ = _ai_backend_status()
    if not ready:
        return
    with st.expander("Ask about this chart"):
        with st.form(key=f"mini_form_{key_suffix}", clear_on_submit=False):
            q = st.text_input("Question", key=f"mini_q_{key_suffix}",
                               label_visibility="collapsed", placeholder=placeholder)
            submitted = st.form_submit_button("Ask")
        if submitted and q:
            with st.spinner("Thinking..."):
                try:
                    full_context = f"WHAT THE USER IS LOOKING AT RIGHT NOW:\n{chart_context}"
                    have_full_ingredients = (
                        scenario and data_dir and access_df is not None and occ_df is not None
                        and results_df is not None and report is not None
                    )
                    if have_full_ingredients:
                        from qa import build_static_context, build_entity_context
                        inst = load_instance(data_dir)
                        static_ctx = build_static_context(inst, access_df, occ_df, results_df, report, scenario)
                        entity_ctx = build_entity_context(inst, access_df, occ_df, q + " " + extra_entity_hint)
                        full_context += "\n\n" + static_ctx + (("\n\n" + entity_ctx) if entity_ctx else "")
                    answer = _call_gemini(q, full_context)
                    st.markdown(answer)
                except ImportError:
                    st.error("Run `pip install google-genai` to enable this feature.")
                except Exception as e:
                    st.error(f"Gemini request failed: {e}")


def render_qa(scenario: str, data_dir: str, access_df, occ_df, results_df, report, worst_contract: str | None):
    st.subheader("Ask the Scheduler")
    st.caption("Powered by Gemini — ask why a contract is delayed, what's using a "
               "location, a handover brief for the next shift, etc.")
    backend, ready, creds = _ai_backend_status()
    if not ready:
        if backend == "vertex":
            st.info("Enter your GCP Project ID in the sidebar to enable Vertex AI Q&A.")
        else:
            st.info("Enter a Gemini API key in the sidebar to enable Q&A "
                    "(free key: https://aistudio.google.com/apikey).")
        return

    example_questions = []
    if worst_contract:
        example_questions.append(f"Why is {worst_contract} overrunning?")
    example_questions.append("Which locations are the biggest bottlenecks?")
    example_questions.append("What's the overall status of this schedule?")

    st.caption("Try asking:")
    cols = st.columns(len(example_questions))
    clicked_example = None
    for i, (col, eq) in enumerate(zip(cols, example_questions)):
        if col.button(eq, key=f"ex_{scenario}_{i}", width='stretch'):
            clicked_example = eq

    with st.form(key=f"qa_form_{scenario}", clear_on_submit=False):
        question = st.text_input("Or type your own question", key=f"q_{scenario}",
                                  placeholder="e.g. What should I prioritize this week?")
        ask_clicked = st.form_submit_button("Ask")

    final_question = clicked_example or (question if ask_clicked and question else None)

    if final_question:
        with st.spinner("Thinking..."):
            try:
                from qa import build_static_context, build_entity_context
                inst = load_instance(data_dir)
                static_ctx = build_static_context(inst, access_df, occ_df, results_df, report, scenario)
                entity_ctx = build_entity_context(inst, access_df, occ_df, final_question)
                full_context = static_ctx + ("\n\n" + entity_ctx if entity_ctx else "")
                answer = _call_gemini(final_question, full_context)
                st.markdown(f"**Q: {final_question}**")
                st.markdown(answer)
            except ImportError:
                st.error("Run `pip install google-genai` to enable this feature.")
            except Exception as e:
                st.error(f"Gemini request failed: {e}")


def render_scenario(scenario: str, data_dir: str):
    spinner_msg = (f"Running Scenario {scenario}..." if scenario != "B"
                   else "Running Scenario B (exact optimization -- can take up to a minute)...")
    with st.spinner(spinner_msg):
        access_df, occ_df, results_df, report = run_scenario_cached(data_dir, scenario)

    inst = load_instance(data_dir)
    feasible = report["feasible"]
    ss = report["soft_scores"]

    disp = results_df.copy()
    disp["overrun_days"] = disp["overrun_days"].fillna(0)
    worst_contract = None
    if (disp["overrun_days"] > 0).any():
        worst_contract = disp.loc[disp["overrun_days"].idxmax(), "contract_number"]

    render_briefing(inst, results_df, occ_df, report, scenario)
    render_qa(scenario, data_dir, access_df, occ_df, results_df, report, worst_contract)
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Feasible", "Yes" if feasible else "No",
              help="Whether every hard rule (safety buffers, capacity limits, weekly allocation "
                   "caps, etc.) is satisfied. A 'No' means this submission wouldn't pass judging as-is.")
    c2.metric("Hard violations", len(report["hard_violations"]),
              help="Count of broken hard rules. Must be zero for a valid submission.")
    c3.metric("Overrun (days)", ss["overrun_days_total"],
              help="Total days, summed across all contracts, that finish after their planned "
                   "completion date.")
    c4.metric("Excess access-nights", ss["excess_access_nights_total"],
              help="Nights where more activities used a location than its normal capacity allows. "
                   "Only permitted (and only scored, not forbidden) in Scenarios B and C.")
    c5.metric("ECLO nights", ss["eclo_nights_total"],
              help="Early Closure / Late Opening — extending a night's work by closing the track "
                   "earlier or reopening it later than usual. Buys 1.5x normal progress per night, "
                   "but cuts into passenger service hours, so it's used sparingly.")

    st.metric("Objective score (lower is better)", report["objective_score"],
              help="The overall penalty score for this schedule. Combines schedule delay (weighted "
                   "by contract priority), excess capacity used, and ECLO nights into one number — "
                   "lower is better, zero is perfect.")

    if not feasible:
        st.error(f"{len(report['hard_violations'])} hard violation(s) found:")
        vdf = pd.DataFrame(report["hard_violations"])
        st.dataframe(vdf, width='stretch', hide_index=True)

    st.subheader("Contract completion summary")
    try:
        st.dataframe(
            disp.style.background_gradient(subset=["overrun_days"], cmap="Reds"),
            width='stretch', hide_index=True,
        )
    except ImportError:
        st.dataframe(disp, width='stretch', hide_index=True)
    render_inline_ai_note(
        f"completion_{scenario}",
        f"SCENARIO {scenario} CONTRACT COMPLETION SUMMARY:\n" + disp.to_string(index=False),
        placeholder="e.g. Which contracts need my attention first?",
        scenario=scenario, data_dir=data_dir, access_df=access_df, occ_df=occ_df,
        results_df=results_df, report=report,
    )

    render_timeline(scenario, inst, access_df, occ_df, results_df, report, data_dir)

    st.subheader("Capacity hotspots (top 10 most-contested locations)")
    hotspot_raw = (
        occ_df.groupby("location_id")["activity_id"].nunique()
        .sort_values(ascending=False).head(10)
    )
    hotspot_context = "\n".join(
        f"{humanize_location(inst, loc)} [{loc}]: {n} activities (nominal capacity {inst.supply.get(loc, '?')})"
        for loc, n in hotspot_raw.items()
    )
    hotspot = hotspot_raw.copy()
    hotspot.index = [humanize_location(inst, loc) for loc in hotspot.index]
    hotspot = hotspot.rename("distinct activities using this location")
    st.bar_chart(hotspot)
    render_inline_ai_note(
        f"hotspot_{scenario}",
        f"SCENARIO {scenario} TOP 10 CAPACITY HOTSPOTS:\n{hotspot_context}",
        placeholder="e.g. Why is this location so contested?",
        scenario=scenario, data_dir=data_dir, access_df=access_df, occ_df=occ_df,
        results_df=results_df, report=report,
    )

    st.subheader("Download submission files")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("SCHEDULE_ACCESS.csv", csv_bytes(access_df),
                        file_name=f"SCHEDULE_ACCESS_{scenario}.csv", width='stretch')
    d2.download_button("SCHEDULE_OCCUPANCY.csv", csv_bytes(occ_df),
                        file_name=f"SCHEDULE_OCCUPANCY_{scenario}.csv", width='stretch')
    d3.download_button("RESULTS.csv", csv_bytes(results_df),
                        file_name=f"RESULTS_{scenario}.csv", width='stretch')
    d4.download_button("All 3 + report (.zip)", make_zip(access_df, occ_df, results_df, report),
                        file_name=f"scenario_{scenario}_submission.zip", width='stretch')


if "has_run" not in st.session_state:
    st.session_state["has_run"] = False

if run_clicked:
    try:
        if uploaded and not use_sample:
            st.session_state["data_dir"] = materialize_instance_dir(uploaded)
        else:
            st.session_state["data_dir"] = str(Path(__file__).parent / "data")
        st.session_state["has_run"] = True
    except FileNotFoundError as e:
        st.error(str(e))
        st.session_state["has_run"] = False

if st.session_state["has_run"]:
    data_dir = st.session_state["data_dir"]
    try:
        tabs_needed = [s for s, flag in [("A", run_a), ("B", run_b), ("C", run_c)] if flag]
        if not tabs_needed:
            st.warning("Select at least one scenario to run.")
        else:
            if len(tabs_needed) >= 2:
                render_comparison(tabs_needed, data_dir)
            tabs = st.tabs([f"Scenario {s}" for s in tabs_needed])
            for tab, s in zip(tabs, tabs_needed):
                with tab:
                    render_scenario(s, data_dir)
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload your 8 instance CSVs (or use the bundled sample) in the sidebar, "
            "pick which scenarios to run, then click **Run scheduler**.")
    st.markdown("""
    **Expected files:** `01_LINES.csv`, `02_STATIONS.csv`, `03_SECTORS.csv`,
    `04_LOCATION_SUPPLY.csv`, `05_BUFFER_LOCATION.csv`, `06_PARAMETERS.csv`,
    `07_PROJECT_DETAILS.csv`, `08_ACTIVITY_DETAILS.csv` -- or a single `.zip`
    containing all of them.
    """)
