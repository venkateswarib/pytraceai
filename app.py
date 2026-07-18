"""
PyTraceAi - app.py

Streamlit demo application for AI-powered PySpark lineage extraction.
Run with:  streamlit run app.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, ".")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PyTraceAi — PySpark Lineage",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Confidence color classes */
.conf-high { color:#2DC653; font-weight:700; }
.conf-med  { color:#F4A261; font-weight:700; }
.conf-low  { color:#E63946; font-weight:700; }

/* Source badges */
.badge {
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:0.72em; font-weight:700; color:white; letter-spacing:.04em;
}
.badge-both    { background:#2DC653; }
.badge-ast     { background:#3A86FF; }
.badge-llm     { background:#9B59B6; }
.badge-partial { background:#F4A261; color:#222; }

/* Section headers */
.section-hdr {
    font-size:0.78em; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#636E72; margin-bottom:4px;
}

/* Summary bar */
.summary-bar {
    background:#F0F4F8; border:1px solid #D0D9E6; border-radius:8px;
    padding:14px 20px; margin-bottom:16px;
}

/* Review alert row */
.review-row {
    background:#FFF5F5; border-left:4px solid #E63946;
    padding:8px 12px; margin:4px 0; border-radius:0 4px 4px 0;
    font-size:0.88em;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPTS = {
    "claims_etl.py  —  Clean PySpark (baseline)":       "sample_scripts/claims_etl.py",
    "fraud_detection.py  —  Embedded SQL":               "sample_scripts/fraud_detection.py",
    "premium_calc.py  —  Config-driven (hard patterns)": "sample_scripts/premium_calc.py",
}

SCRIPT_STORY = {
    "sample_scripts/claims_etl.py":
        "Standard PySpark with literal table names. AST and LLM fully agree. Expect confidence ~100%.",
    "sample_scripts/fraud_detection.py":
        "Joins live inside a spark.sql() string — invisible to the AST. LLM recovers them. Expect confidence ~87%.",
    "sample_scripts/premium_calc.py":
        "Config-dict table names, JDBC reads, helper functions, f-strings. AST finds 0 sources. LLM recovers all. Expect confidence ~69%.",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _conf_color(c: float) -> str:
    if c >= 0.95: return "#2DC653"
    if c >= 0.70: return "#F4A261"
    return "#E63946"

def _conf_html(c: float) -> str:
    cls = "conf-high" if c >= 0.95 else ("conf-med" if c >= 0.70 else "conf-low")
    return f'<span class="{cls}">{c*100:.1f}%</span>'

def _source_html(s: str) -> str:
    mapping = {
        "both":         ("AST + LLM", "badge-both"),
        "ast_only":     ("AST only",  "badge-ast"),
        "llm_only":     ("LLM only",  "badge-llm"),
        "both_partial": ("Partial",   "badge-partial"),
    }
    label, cls = mapping.get(s, (s, "badge-ast"))
    return f'<span class="badge {cls}">{label}</span>'

def _review_icon(needs: bool) -> str:
    return "⚠️" if needs else "✅"

def _severity(item: dict) -> str:
    conf    = item.get("confidence", 1.0)
    section = item.get("section", "")
    source  = item.get("source", "both")
    # Block: writing to wrong target is dangerous, or very low confidence
    if section == "targets" or conf < 0.55:
        return "block"
    # Investigate: single-source extractions (LLM or AST alone)
    if source in ("llm_only", "ast_only"):
        return "investigate"
    # Info: both sources partially agree
    return "info"

def _suggested_action(item: dict) -> str:
    source  = item.get("source", "both")
    section = item.get("section", "")
    conf    = item.get("confidence", 1.0)
    dataset = str(item.get("dataset") or item.get("join_key") or "").lower()
    if conf < 0.55:
        return "Do not promote to production without a manual lineage trace."
    if "jdbc" in dataset:
        return "Confirm database credentials and query scope with the DB owner."
    if source == "llm_only" and section == "targets":
        return "Confirm write destination with data governance before promoting to production."
    if source == "llm_only" and section == "sources":
        return "Verify table exists in data catalog — resolved from a dynamic pattern, not confirmed by AST."
    if source == "ast_only" and section == "sources":
        return "Confirm this table is still actively read — LLM did not detect it as a meaningful source."
    if source == "llm_only" and section == "joins":
        return "Have a DBA validate this join — detected inside a SQL string, not PySpark syntax."
    if source == "ast_only" and section == "joins":
        return "Confirm this join is still in use — LLM did not detect it."
    if source == "both_partial":
        return "Minor discrepancy between sources — spot-check recommended."
    return "Review manually and confirm with the pipeline author."

def _merged_path(script_path: str) -> Path:
    return Path(f"outputs/{Path(script_path).stem}_merged.json")

def _ast_path(script_path: str) -> Path:
    return Path(f"outputs/{Path(script_path).stem}_ast.json")

def _llm_path(script_path: str) -> Path:
    return Path(f"outputs/{Path(script_path).stem}_llm.json")

def _graph_path(script_path: str) -> Path:
    return Path(f"outputs/{Path(script_path).stem}_graph.png")

def _load_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 PyTraceAi")
    st.caption("AI-powered PySpark data lineage extraction")
    st.divider()

    selected_name = st.selectbox(
        "PySpark Script",
        list(SCRIPTS.keys()),
        help="Select a sample script to analyse",
    )
    script_path = SCRIPTS[selected_name]

    st.info(SCRIPT_STORY[script_path], icon="💡")

    api_key = st.text_input(
        "OpenRouter API Key",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
        help="Required for the LLM extraction step",
    )

    force_rerun = st.checkbox(
        "Force re-extraction",
        value=False,
        help="Re-run even if cached outputs exist",
    )

    run_btn = st.button("▶  Run Extraction", type="primary", use_container_width=True)

    st.divider()
    with st.expander("How it works", expanded=False):
        st.markdown("""
**Step 1 — AST Parser**
Static code analysis. Deterministic and exact. Breaks on dynamic patterns.

**Step 2 — LLM (Claude)**
Reads the script like a human. Recovers config-driven tables, embedded SQL, helper functions.

**Step 3 — Merge + Score**
Compare both outputs. `100%` = both agree. `75%` = AST only. `60%` = LLM only.

**Step 4 — Graph**
Visualise data flow with confidence-coded edges.
""")

    with st.expander("View source code", expanded=False):
        code = Path(script_path).read_text(encoding="utf-8")
        st.code(code, language="python")


# ── Run extraction ────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Enter your OpenRouter API key in the sidebar to continue.")
        st.stop()

    os.environ["OPENROUTER_API_KEY"] = api_key
    stem = Path(script_path).stem

    needs_run = force_rerun or not _merged_path(script_path).exists()

    if needs_run:
        with st.status("Extracting lineage...", expanded=True) as status:
            st.write("**Step 1** — Running AST parser...")
            from ast_parser import parse_file
            ast_result = parse_file(script_path)
            st.write(f"  Found {len(ast_result['sources'])} source(s), "
                     f"{len(ast_result['targets'])} target(s), "
                     f"{len(ast_result['joins'])} join(s)")

            st.write("**Step 2** — Calling LLM (Claude via OpenRouter)...")
            from extractor import extract_lineage_from_file
            merged = extract_lineage_from_file(script_path, save_output=True)
            st.write(f"  LLM returned {len(merged.get('_llm_raw', {}).get('sources', []))} source(s)")

            st.write("**Step 3** — Building lineage graph...")
            from graph import render_graph_from_file
            render_graph_from_file(str(_merged_path(script_path)))
            st.write(f"  Graph saved")

            status.update(label="Extraction complete!", state="complete")
    else:
        st.toast("Using cached outputs. Check 'Force re-extraction' to re-run.", icon="📦")

# ── Load and display results ──────────────────────────────────────────────────
merged = _load_json(_merged_path(script_path))
ast_raw = _load_json(_ast_path(script_path))
llm_raw = _load_json(_llm_path(script_path))

if merged is None:
    st.markdown("""
## Welcome to PyTraceAi

Select a script in the sidebar and click **Run Extraction** to begin.

### The demo story

| Script | Pattern | Expected Confidence |
|---|---|---|
| `claims_etl.py` | Clean PySpark | ~100% — AST + LLM fully agree |
| `fraud_detection.py` | Embedded SQL joins | ~87% — AST misses SQL joins |
| `premium_calc.py` | Config-driven tables | ~69% — AST finds 0 sources |

The confidence score tells you exactly what was statically verified vs AI-inferred.
""")
    st.stop()

# ── Summary bar ───────────────────────────────────────────────────────────────
overall           = merged.get("overall_confidence", 0)
review_items_list = merged.get("needs_review", [])
n_review          = len(review_items_list)
n_src             = len(merged.get("sources", []))

# Session state for review checkboxes — computed before tabs so badge updates live
script_key = Path(script_path).stem
if "reviewed_items" not in st.session_state:
    st.session_state.reviewed_items = set()
n_unreviewed = sum(
    1 for i in range(n_review)
    if f"{script_key}_{i}" not in st.session_state.reviewed_items
)
n_tgt = len(merged.get("targets", []))
n_joins = len(merged.get("joins", []))
n_renames = len(merged.get("column_renames", []))

st.markdown(f"## {Path(script_path).name}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall Confidence",
          f"{overall*100:.1f}%",
          delta="High" if overall >= 0.95 else ("Medium" if overall >= 0.70 else "Needs Review"),
          delta_color="normal" if overall >= 0.70 else "inverse")
c2.metric("Sources Found",    n_src)
c3.metric("Targets Found",    n_tgt)
c4.metric("Joins Detected",   n_joins)
c5.metric("Needs Review",     n_review,
          delta_color="inverse" if n_review > 0 else "normal")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_graph, tab_lineage, tab_compare, tab_business, tab_review = st.tabs([
    "📊  Lineage Graph",
    "🗄  Sources & Targets",
    "⚖  AST vs LLM",
    "💼  Business Context",
    "✅  All Reviewed" if n_unreviewed == 0 and n_review > 0 else f"⚠  Needs Review ({n_unreviewed})",
])


# ── Tab 1: Graph ──────────────────────────────────────────────────────────────
with tab_graph:
    gp = _graph_path(script_path)
    if gp.exists():
        st.image(str(gp), use_container_width=True)
    else:
        st.info("Run extraction to generate the graph.")

    # OpenLineage download
    import json as _json
    from openlineage_emitter import to_openlineage
    ol_event = to_openlineage(merged, Path(script_path).stem)
    st.download_button(
        label="⬇ Download OpenLineage JSON",
        data=_json.dumps(ol_event, indent=2),
        file_name=f"{Path(script_path).stem}_openlineage.json",
        mime="application/json",
        help="OpenLineage-compliant RunEvent — import into Marquez, DataHub, or Atlan",
    )

    with st.expander("Join details", expanded=True):
        joins = merged.get("joins", [])
        if not joins:
            st.write("No joins detected.")
        else:
            for j in joins:
                key = j.get("join_key", "?")
                key_str = ", ".join(key) if isinstance(key, list) else str(key)
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(
                    f"**{j.get('left','?')}** `{j.get('join_type','?').upper()}` "
                    f"**{j.get('right','?')}**  —  ON `{key_str}`  "
                    f"{_source_html(j.get('source','both'))}",
                    unsafe_allow_html=True,
                )
                col_b.markdown(_conf_html(j.get("confidence", 1.0)), unsafe_allow_html=True)
                if j.get("description"):
                    st.caption(j["description"])


# ── Tab 2: Sources & Targets ──────────────────────────────────────────────────
with tab_lineage:
    left, right = st.columns(2)

    with left:
        st.markdown('<p class="section-hdr">Sources</p>', unsafe_allow_html=True)
        for src in merged.get("sources", []):
            with st.container(border=True):
                r1, r2 = st.columns([4, 1])
                r1.markdown(
                    f"**{src['dataset']}**  "
                    f"{_source_html(src.get('source','both'))}",
                    unsafe_allow_html=True,
                )
                r2.markdown(_conf_html(src.get("confidence", 1.0)),
                            unsafe_allow_html=True)
                if src.get("method"):
                    st.caption(f"Method: `{src['method']}`")
                if src.get("description"):
                    st.caption(src["description"])

    with right:
        st.markdown('<p class="section-hdr">Targets</p>', unsafe_allow_html=True)
        for tgt in merged.get("targets", []):
            with st.container(border=True):
                r1, r2 = st.columns([4, 1])
                r1.markdown(
                    f"**{tgt['dataset']}**  "
                    f"{_source_html(tgt.get('source','both'))}",
                    unsafe_allow_html=True,
                )
                r2.markdown(_conf_html(tgt.get("confidence", 1.0)),
                            unsafe_allow_html=True)
                if tgt.get("method"):
                    st.caption(f"Method: `{tgt['method']}`")
                if tgt.get("description"):
                    st.caption(tgt["description"])

    st.divider()
    st.markdown('<p class="section-hdr">Column Renames</p>', unsafe_allow_html=True)
    renames = merged.get("column_renames", [])
    if renames:
        rows = []
        for r in renames:
            rows.append({
                "Old Name":        r["old_name"],
                "New Name":        r["new_name"],
                "Confidence":      r.get("confidence", 1.0),
                "Source":          r.get("source", "both"),
                "Business Reason": r.get("business_reason", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.NumberColumn(format="%.2f"),
                "Old Name":   st.column_config.TextColumn(width="small"),
                "New Name":   st.column_config.TextColumn(width="small"),
            },
            hide_index=True,
        )
    else:
        st.caption("No column renames detected.")


# ── Tab 3: AST vs LLM ────────────────────────────────────────────────────────
with tab_compare:
    st.caption("Side-by-side comparison of what each source found independently.")

    col_ast, col_llm = st.columns(2)

    with col_ast:
        st.markdown("### 🔵 AST Parser")
        st.caption("Static analysis — deterministic, exact, no interpretations.")

        if ast_raw:
            src_count = len(ast_raw.get("sources", []))
            tgt_count = len(ast_raw.get("targets", []))
            jn_count  = len(ast_raw.get("joins", []))

            st.metric("Sources found", src_count)
            st.metric("Targets found", tgt_count)
            st.metric("Joins found",   jn_count)

            st.markdown("**Sources:**")
            if ast_raw.get("sources"):
                for s in ast_raw["sources"]:
                    st.code(s["dataset"], language=None)
            else:
                st.warning("No sources found by AST.", icon="⚠️")

            st.markdown("**Targets:**")
            if ast_raw.get("targets"):
                for t in ast_raw["targets"]:
                    st.code(t["dataset"], language=None)
            else:
                st.warning("No targets found by AST.", icon="⚠️")

            st.markdown("**Joins:**")
            if ast_raw.get("joins"):
                for j in ast_raw["joins"]:
                    key = j.get("join_key","?")
                    key_str = ", ".join(key) if isinstance(key, list) else str(key)
                    st.code(f"{j['left']} x {j['right']} ON {key_str} ({j['join_type']})",
                            language=None)
            else:
                st.warning("No joins found by AST.", icon="⚠️")

            if ast_raw.get("sql_blocks"):
                st.markdown("**Embedded SQL:**")
                for sql in ast_raw["sql_blocks"]:
                    st.code(sql, language="sql")
        else:
            st.info("Run extraction to see AST output.")

    with col_llm:
        st.markdown("### 🟣 LLM (Claude)")
        st.caption("Semantic analysis — reads intent, resolves dynamic patterns, adds business context.")

        if llm_raw:
            src_count = len(llm_raw.get("sources", []))
            tgt_count = len(llm_raw.get("targets", []))
            jn_count  = len(llm_raw.get("joins", []))

            st.metric("Sources found", src_count)
            st.metric("Targets found", tgt_count)
            st.metric("Joins found",   jn_count)

            st.markdown("**Sources:**")
            for s in llm_raw.get("sources", []):
                st.code(s["dataset"], language=None)
                if s.get("description"):
                    st.caption(s["description"])

            st.markdown("**Targets:**")
            for t in llm_raw.get("targets", []):
                st.code(t["dataset"], language=None)
                if t.get("description"):
                    st.caption(t["description"])

            st.markdown("**Joins:**")
            for j in llm_raw.get("joins", []):
                key = j.get("join_key","?")
                key_str = ", ".join(key) if isinstance(key, list) else str(key)
                st.code(f"{j['left']} x {j['right']} ON {key_str} ({j['join_type']})",
                        language=None)
                if j.get("description"):
                    st.caption(j["description"])
        else:
            st.info("Run extraction to see LLM output.")

    # Delta callout
    if ast_raw and llm_raw:
        ast_src = {s["dataset"] for s in ast_raw.get("sources", [])}
        llm_src = {s["dataset"] for s in llm_raw.get("sources", [])}
        llm_only_src = llm_src - ast_src
        ast_only_src = ast_src - llm_src

        if llm_only_src or ast_only_src:
            st.divider()
            st.markdown("#### Gap Analysis")
            if llm_only_src:
                st.error(
                    f"**LLM recovered {len(llm_only_src)} source(s) the AST missed** "
                    f"(hidden behind dynamic patterns):\n\n"
                    + "\n".join(f"- `{s}`" for s in sorted(llm_only_src)),
                    icon="🤖",
                )
            if ast_only_src:
                st.warning(
                    f"**AST found {len(ast_only_src)} source(s) the LLM missed:**\n\n"
                    + "\n".join(f"- `{s}`" for s in sorted(ast_only_src)),
                    icon="🔵",
                )
        else:
            st.success("AST and LLM found the same sources — high confidence extraction.", icon="✅")


# ── Tab 4: Business Context ───────────────────────────────────────────────────
with tab_business:
    summary = merged.get("business_summary", "")
    if summary:
        st.markdown("### Business Summary")
        st.info(summary, icon="📋")
    else:
        st.caption("No business summary available.")

    transforms = merged.get("transformations", [])
    if transforms:
        st.markdown("### Transformations")
        type_order = ["filter", "aggregate", "derive", "cast", "window"]
        grouped: dict[str, list] = {}
        for t in transforms:
            tp = t.get("type", "other")
            grouped.setdefault(tp, []).append(t["description"])

        for tp in type_order + [k for k in grouped if k not in type_order]:
            if tp not in grouped:
                continue
            icons = {"filter":"🔽","aggregate":"∑","derive":"✏️","cast":"🔄","window":"🪟"}
            icon = icons.get(tp, "•")
            st.markdown(f"**{icon} {tp.title()}**")
            for desc in grouped[tp]:
                st.markdown(f"  - {desc}")

    sql = merged.get("sql_blocks", [])
    if sql:
        st.markdown("### Embedded SQL")
        for block in sql:
            st.code(block, language="sql")


# ── Tab 5: Needs Review ───────────────────────────────────────────────────────
with tab_review:
    if not review_items_list:
        st.success("All items verified by both AST and LLM. Nothing needs review.", icon="✅")
    else:
        reviewed_count = n_review - n_unreviewed
        st.progress(
            reviewed_count / n_review,
            text=f"{reviewed_count} of {n_review} items resolved",
        )

        blocks       = [(i, it) for i, it in enumerate(review_items_list) if _severity(it) == "block"]
        investigates = [(i, it) for i, it in enumerate(review_items_list) if _severity(it) == "investigate"]
        infos        = [(i, it) for i, it in enumerate(review_items_list) if _severity(it) == "info"]

        def _render_tier(tier_items, icon, label):
            if not tier_items:
                return
            st.markdown(f"#### {icon} {label} — {len(tier_items)} item(s)")
            for idx, item in tier_items:
                item_key = f"{script_key}_{idx}"
                is_done  = item_key in st.session_state.reviewed_items

                section = item.get("section", "?")
                dataset = (
                    item.get("dataset")
                    or item.get("join_key")
                    or item.get("old_name", "?")
                )
                conf   = item.get("confidence", 0)
                source = item.get("source", "?")
                desc   = (
                    item.get("description")
                    or item.get("business_reason")
                    or ""
                )
                action = _suggested_action(item)

                with st.container(border=True):
                    col_cb, col_body = st.columns([0.5, 9.5])

                    with col_cb:
                        checked = st.checkbox(
                            "Done", value=is_done,
                            key=f"cb_{item_key}",
                            label_visibility="collapsed",
                        )
                        if checked:
                            st.session_state.reviewed_items.add(item_key)
                        else:
                            st.session_state.reviewed_items.discard(item_key)

                    with col_body:
                        r1, r2 = st.columns([5, 1])
                        r1.markdown(
                            f"**[{section.upper()}]** `{dataset}`  "
                            f"{_source_html(source)}",
                            unsafe_allow_html=True,
                        )
                        r2.markdown(_conf_html(conf), unsafe_allow_html=True)

                        st.markdown(
                            f"<div style='background:#EFF6FF;border-left:3px solid #3A86FF;"
                            f"padding:6px 10px;border-radius:0 4px 4px 0;font-size:0.85em;margin:4px 0'>"
                            f"<strong>Suggested action:</strong> {action}</div>",
                            unsafe_allow_html=True,
                        )
                        if desc:
                            st.caption(desc[:120] + "..." if len(desc) > 120 else desc)

        _render_tier(blocks,       "🔴", "Block — resolve before production")
        _render_tier(investigates, "🟡", "Investigate — verify with data owner")
        _render_tier(infos,        "🟢", "Informational — low risk, spot-check")
