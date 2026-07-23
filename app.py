"""
PyTraceAi - app.py

Streamlit demo application for AI-powered PySpark lineage extraction.
Run with:  streamlit run app.py
"""

import functools
import io
import json
import os
import sys
import tokenize
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

/* ── Hero banner (sidebar) ── */
.pytraceai-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #2563EB 100%);
    border-radius: 10px;
    padding: 16px 16px 14px 16px;
    margin-bottom: 4px;
}
.pytraceai-hero h1 {
    margin: 0 0 2px 0;
    font-size: 1.5em;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}
.pytraceai-hero .sub {
    font-size: 0.8em;
    color: #93C5FD;
    margin: 0;
    line-height: 1.35;
}
.pytraceai-hero .tagline {
    display: inline-block;
    margin-top: 8px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 14px;
    padding: 3px 10px;
    font-size: 0.64em;
    color: #BFDBFE;
    letter-spacing: 0.03em;
}


/* Prominent download buttons */
div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #3A86FF 0%, #2563EB 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95em !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
    border: none !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(58,134,255,0.4) !important;
    transition: all 0.2s !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    box-shadow: 0 4px 14px rgba(58,134,255,0.55) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPTS = {
    "claims_etl.py  —  Clean PySpark (AST wins)":                  "sample_scripts/claims_etl.py",
    "fraud_detection.py  —  Encoded formula (blind spot #1)":       "sample_scripts/fraud_detection.py",
    "premium_calc.py  —  Runtime config (blind spot #2)":           "sample_scripts/premium_calc.py",
}

SCRIPT_STORY = {
    "sample_scripts/claims_etl.py":
        "Clean PySpark with literal table names and explicit transformations. "
        "AST and LLM fully agree — confidence ~100%. "
        "Baseline: when code is transparent, AST is fast, exact, and sufficient.",
    "sample_scripts/fraud_detection.py":
        "Sources and target are plain text — AST finds them. "
        "But the core underwriting formula (Loss_Ratio derivation) is base64-encoded. "
        "AST sees an opaque string; LLM decodes it and recovers the column-level transformation.",
    "sample_scripts/premium_calc.py":
        "Transformations are explicit — AST reads them fine. "
        "But table names come from os.environ and config dicts at runtime. "
        "AST resolves only 1 of 4 sources; LLM infers the full table list from context.",
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

@functools.lru_cache(maxsize=8)
def _docstring_lines(script_path: str) -> frozenset[int]:
    """Line numbers that fall inside a multi-line string literal (module/
    function docstrings). Excluded from code-reference search — prose in a
    docstring can coincidentally contain a search term (e.g. a column name
    mentioned in a comment) and must not be mistaken for the real code line."""
    try:
        source = Path(script_path).read_text(encoding="utf-8")
    except Exception:
        return frozenset()
    lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
                lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    return frozenset(lines)


def _find_join_ref(
    lines: list[str], docstring_lines: frozenset[int],
    key_list: list[str], left_name: str, right_name: str,
) -> list[tuple[int, str]]:
    """Locate the specific .join(...) call this item describes, then return
    its 'on=' clause line. A plain substring search on the join key alone is
    unreliable — the same column name often also appears in an unrelated SQL
    string or config dict elsewhere in the script (e.g. a JDBC query that
    selects the same columns). Anchoring on the nearest '.join(' call site
    and scoring candidate blocks by how many of {left df, right df, key(s)}
    they contain disambiguates the real join from a coincidental text match."""
    call_lines = [i for i, line in enumerate(lines, 1)
                  if i not in docstring_lines and ".join(" in line]

    def _is_exact_receiver(line: str) -> bool:
        """True if `left_name` is literally the receiver right before
        '.join(' on this line — e.g. 'rated_df.join(' or
        'x = rated_df.join('. Two different .join() calls can share the
        same left/right/key text somewhere in their block (e.g. one call's
        assignment target is another call's receiver name); this is the
        one unambiguous signal that picks the correct call site."""
        idx = line.find(".join(")
        return bool(left_name) and idx != -1 and line[:idx].rstrip().endswith(left_name)

    exact_calls = [i for i in call_lines if _is_exact_receiver(lines[i - 1])]
    candidate_starts = exact_calls or call_lines

    best_block, best_score = None, -1
    for start in candidate_starts:
        end = min(start + 7, len(lines))
        block_text = " ".join(lines[start - 1:end])
        score = sum([
            bool(left_name) and left_name in block_text,
            bool(right_name) and right_name in block_text,
        ]) + sum(1 for k in key_list if k and k in block_text)
        if start in exact_calls:
            score += 10  # exact receiver match beats any coincidental overlap
        if score > best_score:
            best_score, best_block = score, (start, end)
    if best_block is None:
        return []
    start, end = best_block
    for i in range(start, end + 1):
        if i in docstring_lines:
            continue
        stripped = lines[i - 1].strip()
        if "on=" in stripped or "on ==" in stripped or "on =" in stripped \
                or any(k and k in stripped for k in key_list):
            return [(i, stripped)]
    return [(start, lines[start - 1].strip())]


def _find_code_refs(script_path: str, item: dict) -> list[tuple[int, str]]:
    """Find script lines that reference the item's key identifier.
    Returns up to 3 (line_no, stripped_line) tuples."""
    try:
        lines = Path(script_path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    docstring_lines = _docstring_lines(script_path)

    section = item.get("section", "")
    stop_at_first_hit = True
    if section in ("sources", "targets"):
        dataset = item.get("dataset", "")
        if dataset.startswith("jdbc:") and "/" in dataset:
            # A jdbc dataset is assembled from two independent script
            # locations — the connection URL (often in a config dict) and
            # the table name (often in a separate SQL query string). Both
            # are needed to explain where the value came from, so search
            # for both instead of stopping once one of them is found.
            url_part, _, table_part = dataset.rpartition("/")
            candidates = [table_part, url_part]
            stop_at_first_hit = False
        elif "." in dataset:
            candidates = [dataset, dataset.split(".", 1)[1]]
        else:
            candidates = [dataset]
    elif section == "joins":
        key = item.get("join_key", "")
        key_list = key if isinstance(key, list) else ([key] if key else [])
        return _find_join_ref(
            lines, docstring_lines, key_list,
            item.get("left", ""), item.get("right", ""),
        )
    elif section == "column_renames":
        candidates = [item.get("old_name", ""), item.get("new_name", "")]
    else:
        candidates = []

    seen: set[tuple] = set()
    hits: list[tuple[int, str]] = []
    for term in candidates:
        if not term:
            continue
        found_this_term = False
        for i, line in enumerate(lines, 1):
            if i in docstring_lines:
                continue
            stripped = line.strip()
            if term in stripped and not stripped.startswith("#"):
                entry = (i, stripped)
                if entry not in seen:
                    seen.add(entry)
                    hits.append(entry)
                    found_this_term = True
                    if len(hits) >= 3:
                        return hits
        if found_this_term and stop_at_first_hit:
            break
    return hits


def _item_label(item: dict, section: str) -> str:
    if section == "joins":
        key = item.get("join_key", "?")
        key_str = ", ".join(key) if isinstance(key, list) else str(key)
        return f"{item.get('left','?')} ⋈ {item.get('right','?')}  on {key_str}"
    return item.get("dataset", "?")


def _unified_compare_df(script_path: str, ast_raw: dict, llm_raw: dict, merged: dict) -> pd.DataFrame:
    """
    One row per finding, ordered by source line — Code | AST Found | LLM Found.
    A single table instead of separate per-section grids plus a bolted-on
    'hidden logic' box: the code snippet itself carries the story. For a
    plain literal, the snippet is the read/write/join call and both columns
    agree. For an encoded payload, the snippet IS the opaque base64 blob —
    AST's column can only name the bare node type, LLM's column has the
    actual decoded meaning. Seeing all three side by side per row is what
    makes the blind spot visible, without a separate explanatory section.

    Rows are built from the already-merged (post-scoring) data, not
    re-derived from raw ast_raw/llm_raw — extractor.py's merge already
    resolves multi-key vs. split-key join phrasing, sub-key dedup, and
    confidence/needs_review. Re-matching independently here (an earlier
    version of this function did) could disagree with the merge's own
    verdict — e.g. treating an LLM join the merge correctly matched as
    unmatched, because its raw join_key shape didn't equal the AST one.
    """
    rows: list[dict] = []

    def add_merged_section(section: str, items: list[dict]):
        for item in items:
            src = item.get("source", "both")
            if section == "column_renames":
                label = f"{item.get('old_name','?')} → {item.get('new_name','?')}"
                fake = {"section": "column_renames",
                        "old_name": item.get("old_name", ""), "new_name": item.get("new_name", "")}
            elif section == "joins":
                label = _item_label(item, "joins")
                fake = {"section": "joins", "join_key": item.get("join_key", ""),
                        "left": item.get("left", ""), "right": item.get("right", "")}
            else:
                label = _item_label(item, section)
                fake = {"section": section, "dataset": item.get("dataset", "")}

            refs = _find_code_refs(script_path, fake)
            if refs:
                sort_line   = refs[0][0]  # earliest matching line — sort position only
                display_line = ", ".join(str(ln) for ln, _ in refs)
                snippet      = " | ".join(f"L{ln}: {snip}" for ln, snip in refs) if len(refs) > 1 else refs[0][1]
            else:
                sort_line, display_line, snippet = None, None, "—"

            ast_cell = label if src in ("both", "ast_only", "both_partial") else "🚫 not found"
            if src in ("both", "llm_only", "both_partial"):
                desc = item.get("description") or item.get("business_reason") or ""
                llm_cell = f"{label} — {desc}" if desc else label
            else:
                llm_cell = "—"

            rows.append({
                "sort_line": sort_line, "line": display_line, "snippet": snippet,
                "ast": ast_cell, "llm": llm_cell,
                "confidence": f"{item.get('confidence', 1.0) * 100:.0f}%",
                "review": "⚠️ Y" if item.get("needs_review") else "✅ N",
            })

    add_merged_section("sources",        merged.get("sources", []))
    add_merged_section("targets",        merged.get("targets", []))
    add_merged_section("joins",          merged.get("joins", []))
    add_merged_section("column_renames", merged.get("column_renames", []))

    # Filters — AST reads the raw boolean condition (plain syntax, not a
    # blind spot); paired positionally against the LLM's filter-type
    # transformations, since neither side has a natural join key.
    llm_filters = [t for t in merged.get("transformations", []) if t.get("type") == "filter"]
    for i, f in enumerate(ast_raw.get("filters", [])):
        d = llm_filters[i] if i < len(llm_filters) else None
        rows.append({
            "sort_line":  f.get("line"),
            "line":       (str(f["line"]) if f.get("line") is not None else None),
            "snippet":    f'.filter({f["condition"]})',
            "ast":        f'✅ {f["condition"]}',
            "llm":        d["description"] if d else "—",
            "confidence": "100%" if d else "75%",
            "review":     "✅ N" if d else "⚠️ Y",
        })

    # Derived columns and the opaque-logic special case both draw from the
    # same LLM derive-type pool. Opaque logic (encoded, AST cannot read the
    # expression at all) gets first claim on those entries; ordinary
    # .withColumn() calls (AST reads the expression directly, no blind
    # spot) take whatever's left, so a derive-type transformation is never
    # double-counted across both categories.
    derived_all = [t for t in merged.get("transformations", []) if t.get("type") == "derive"]
    opaque = ast_raw.get("opaque_logic", [])
    for i, o in enumerate(opaque):
        d = derived_all[i] if i < len(derived_all) else None
        rows.append({
            "sort_line":  o.get("line"),
            "line":       (str(o["line"]) if o.get("line") is not None else None),
            "snippet":    f'{o["variable"]} = "{o["raw_preview"]}"',
            "ast":        f'⚠️ {o["ast_node"]} — opaque, cannot execute or decode',
            "llm":        d["description"] if d else "Run/re-run extraction to decode this payload.",
            "confidence": "60%" if d else "—",
            "review":     "⚠️ Y",
        })

    remaining_derived = derived_all[len(opaque):]
    for i, dc in enumerate(ast_raw.get("derived_columns", [])):
        d = remaining_derived[i] if i < len(remaining_derived) else None
        expr_preview = dc["expression"][:60] + ("…" if len(dc["expression"]) > 60 else "")
        rows.append({
            "sort_line":  dc.get("line"),
            "line":       (str(dc["line"]) if dc.get("line") is not None else None),
            "snippet":    f'.withColumn("{dc["column"]}", {expr_preview})',
            "ast":        f'✅ {dc["column"]} = {expr_preview}',
            "llm":        d["description"] if d else "—",
            "confidence": "100%" if d else "75%",
            "review":     "✅ N" if d else "⚠️ Y",
        })

    rows.sort(key=lambda r: (r["sort_line"] is None, r["sort_line"] or 0))
    cols = ["Line", "Code", "AST Found", "LLM Found", "Confidence", "Needs Review"]
    return pd.DataFrame([
        {"Line": (r["line"] if r["line"] is not None else "—"), "Code": r["snippet"],
         "AST Found": r["ast"], "LLM Found": r["llm"],
         "Confidence": r["confidence"], "Needs Review": r["review"]}
        for r in rows
    ], columns=cols) if rows else pd.DataFrame(columns=cols)


def _build_lineage_map(merged: dict, script_path: str) -> pd.DataFrame:
    """Build a flat lineage mapping table from the merged lineage dict."""
    rows = []
    targets  = merged.get("targets", [])
    tgt_name = targets[0]["dataset"] if targets else Path(script_path).stem

    def _ref_fields(fake_item: dict) -> tuple[str, str]:
        """Return (line_number_str, code_snippet_str) for a lineage item."""
        refs = _find_code_refs(script_path, fake_item)
        if refs:
            linenos = ", ".join(str(ln) for ln, _ in refs)
            snippet = " | ".join(s[:80] for _, s in refs)
            return linenos, snippet
        src = fake_item.get("source", "both")
        if src == "llm_only":
            return "LLM-inferred", "Not a direct code literal — resolved from dynamic pattern"
        return "", ""

    # 1. Source → target dataset-level rows
    for src in merged.get("sources", []):
        for tgt in (targets if targets else [{"dataset": tgt_name}]):
            nr   = src.get("needs_review", False)
            fake = {"source": src.get("source", "both"), "section": "sources",
                    "confidence": src.get("confidence", 1.0), "dataset": src["dataset"]}
            lineno, snippet = _ref_fields(fake)
            rows.append({
                "Source Dataset":   src["dataset"],
                "Source Column":    "",
                "Target Dataset":   tgt["dataset"],
                "Target Column":    "",
                "Transformation":   src.get("method") or "read",
                "Confidence":       f"{src.get('confidence', 1.0)*100:.1f}%",
                "Needs Review":     "Y" if nr else "N",
                "Suggested Action": _suggested_action(fake) if nr else "",
                "Line Number":      lineno,
                "Code Snippet":     snippet,
            })

    # 2. Join rows
    for j in merged.get("joins", []):
        nr      = j.get("needs_review", False)
        key     = j.get("join_key", "")
        key_str = ", ".join(key) if isinstance(key, list) else str(key)
        fake    = {"source": j.get("source", "both"), "section": "joins",
                   "confidence": j.get("confidence", 1.0), "join_key": key,
                   "left": j.get("left", ""), "right": j.get("right", "")}
        lineno, snippet = _ref_fields(fake)
        rows.append({
            "Source Dataset":   j.get("left", ""),
            "Source Column":    key_str,
            "Target Dataset":   j.get("right", ""),
            "Target Column":    key_str,
            "Transformation":   f"{j.get('join_type','').upper()} JOIN",
            "Confidence":       f"{j.get('confidence', 1.0)*100:.1f}%",
            "Needs Review":     "Y" if nr else "N",
            "Suggested Action": _suggested_action(fake) if nr else "",
            "Line Number":      lineno,
            "Code Snippet":     snippet,
        })

    # 3. Column rename rows — precise column-level lineage
    for r in merged.get("column_renames", []):
        nr      = r.get("needs_review", False)
        reason  = r.get("business_reason", "")
        fake    = {"source": r.get("source", "both"), "section": "column_renames",
                   "confidence": r.get("confidence", 1.0), "old_name": r.get("old_name", "")}
        label   = f"rename — {reason}" if reason else "rename"
        lineno, snippet = _ref_fields(fake)
        rows.append({
            "Source Dataset":   tgt_name,
            "Source Column":    r.get("old_name", ""),
            "Target Dataset":   tgt_name,
            "Target Column":    r.get("new_name", ""),
            "Transformation":   label,
            "Confidence":       f"{r.get('confidence', 1.0)*100:.1f}%",
            "Needs Review":     "Y" if nr else "N",
            "Suggested Action": _suggested_action(fake) if nr else "",
            "Line Number":      lineno,
            "Code Snippet":     snippet,
        })

    # 4. LLM-inferred transformations (description-level)
    for t in merged.get("transformations", []):
        rows.append({
            "Source Dataset":   tgt_name,
            "Source Column":    "",
            "Target Dataset":   tgt_name,
            "Target Column":    "",
            "Transformation":   f"{t.get('type','').title()}: {t.get('description','')}",
            "Confidence":       f"{t.get('confidence', 1.0)*100:.1f}%" if t.get("confidence") else "—",
            "Needs Review":     "N",
            "Suggested Action": "",
            "Line Number":      "",
            "Code Snippet":     "",
        })

    cols = ["Source Dataset", "Source Column", "Target Dataset", "Target Column",
            "Transformation", "Confidence", "Needs Review", "Suggested Action",
            "Line Number", "Code Snippet"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


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
    st.markdown("""
<div class="pytraceai-hero">
  <h1>🔍 PyTraceAi</h1>
  <p class="sub">AI-powered PySpark lineage extraction</p>
  <span class="tagline">AST · LLM · Confidence</span>
</div>
""", unsafe_allow_html=True)
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
### Select a script and click **Run Extraction** to begin.

PyTraceAi combines static AST analysis with an LLM to recover lineage patterns that static parsers cannot reach. Pick each script in order to see the three scenarios:

| Script | AST blind spot | What AST misses | What LLM recovers |
|---|---|---|---|
| `claims_etl.py` | None — clean baseline | Nothing | Nothing new — they agree |
| `fraud_detection.py` | Base64-encoded formula | `Loss_Ratio` derived column | Decodes the underwriting formula |
| `premium_calc.py` | Runtime env/config table names | 3 of 4 source tables | Infers full table list from context |

The **confidence score** tells you exactly what was statically verified vs AI-inferred.
""")
    st.stop()

# ── Summary bar ───────────────────────────────────────────────────────────────
overall           = merged.get("overall_confidence", 0)
review_items_list = merged.get("needs_review", [])
n_review          = len(review_items_list)
n_src             = len(merged.get("sources", []))
n_tgt             = len(merged.get("targets", []))
n_joins           = len(merged.get("joins", []))

st.markdown(f"### `{Path(script_path).name}`")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall Confidence",
          f"{overall*100:.1f}%",
          delta="High" if overall >= 0.95 else ("Medium" if overall >= 0.70 else "Needs Review"),
          delta_color="normal" if overall >= 0.70 else "inverse")
c2.metric("Sources Found",  n_src)
c3.metric("Targets Found",  n_tgt)
c4.metric("Joins Detected", n_joins)
c5.metric("Needs Review",   n_review,
          delta_color="inverse" if n_review > 0 else "normal")

st.divider()

# ── Download buttons ──────────────────────────────────────────────────────────
import json as _json
from openlineage_emitter import to_openlineage

_ol_event    = to_openlineage(merged, Path(script_path).stem)
_mapping_df  = _build_lineage_map(merged, script_path)
_stem        = Path(script_path).stem

dl1, dl2, _ = st.columns([2, 2, 3])
with dl1:
    st.download_button(
        label="⬇  OpenLineage JSON",
        data=_json.dumps(_ol_event, indent=2),
        file_name=f"{_stem}_openlineage.json",
        mime="application/json",
        use_container_width=True,
        help="OpenLineage RunEvent — import into Marquez, DataHub, or Atlan",
        key="dl_ol",
    )
with dl2:
    st.download_button(
        label="⬇  Lineage Mapping CSV",
        data=_mapping_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{_stem}_lineage_mapping.csv",
        mime="text/csv",
        use_container_width=True,
        help="Full source→target mapping with confidence scores and suggested actions",
        key="dl_map",
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_graph, tab_lineage, tab_compare, tab_business, tab_review = st.tabs([
    "📊  Lineage Graph",
    "🗄  Sources & Targets",
    "⚖  AST vs LLM",
    "💼  Business Context",
    f"⚠  Needs Review ({n_review})" if n_review > 0 else "✅  Needs Review",
])


# ── Tab 1: Graph ──────────────────────────────────────────────────────────────
with tab_graph:
    gp = _graph_path(script_path)
    if gp.exists():
        st.image(str(gp), use_container_width=True)
    else:
        st.info("Run extraction to generate the graph.")

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
    st.caption(
        "One row per finding, in script order — the code AST parsed, what AST "
        "concluded, and what the LLM concluded. Where they diverge, that gap "
        "is the blind spot."
    )

    if ast_raw and llm_raw:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sources", f"{len(ast_raw.get('sources',[]))} AST  /  {len(llm_raw.get('sources',[]))} LLM")
        m2.metric("Targets", f"{len(ast_raw.get('targets',[]))} AST  /  {len(llm_raw.get('targets',[]))} LLM")
        m3.metric("Joins",   f"{len(ast_raw.get('joins',[]))} AST  /  {len(llm_raw.get('joins',[]))} LLM")

        df = _unified_compare_df(script_path, ast_raw, llm_raw, merged)
        if df.empty:
            st.caption("Nothing found by either source.")
        else:
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "Line":         st.column_config.TextColumn(width="small"),
                    "Code":         st.column_config.TextColumn(width="large"),
                    "AST Found":    st.column_config.TextColumn(width="medium"),
                    "LLM Found":    st.column_config.TextColumn(width="large"),
                    "Confidence":   st.column_config.TextColumn(width="small"),
                    "Needs Review": st.column_config.TextColumn(width="small"),
                },
            )
    elif ast_raw:
        st.info("LLM output not available yet — run extraction to see the comparison.", icon="🟣")
    else:
        st.info("Run extraction to see the AST vs LLM comparison.")


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
        st.caption(
            f"{n_review} item(s) flagged. Download the **Lineage Mapping CSV** above "
            "for the full list with suggested actions."
        )

        blocks       = [(i, it) for i, it in enumerate(review_items_list) if _severity(it) == "block"]
        investigates = [(i, it) for i, it in enumerate(review_items_list) if _severity(it) == "investigate"]
        infos        = [(i, it) for i, it in enumerate(review_items_list) if _severity(it) == "info"]

        def _render_tier(tier_items, icon, label):
            if not tier_items:
                return
            st.markdown(f"#### {icon} {label} — {len(tier_items)} item(s)")
            for _, item in tier_items:
                section = item.get("section", "?")
                dataset = (
                    item.get("dataset")
                    or item.get("join_key")
                    or item.get("old_name", "?")
                )
                conf   = item.get("confidence", 0)
                source = item.get("source", "?")
                desc   = item.get("description") or item.get("business_reason") or ""
                action = _suggested_action(item)
                refs   = _find_code_refs(script_path, item)

                with st.container(border=True):
                    r1, r2 = st.columns([5, 1])
                    r1.markdown(
                        f"**[{section.upper()}]** `{dataset}`  {_source_html(source)}",
                        unsafe_allow_html=True,
                    )
                    r2.markdown(_conf_html(conf), unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:#EFF6FF;border-left:3px solid #3A86FF;"
                        f"padding:6px 10px;border-radius:0 4px 4px 0;font-size:0.85em;margin:4px 0'>"
                        f"<strong>Suggested action:</strong> {action}</div>",
                        unsafe_allow_html=True,
                    )
                    if refs:
                        for lineno, line_text in refs:
                            st.markdown(
                                f"<span style='font-size:0.75em;color:#636E72;font-family:monospace'>"
                                f"📍 Line {lineno}</span>",
                                unsafe_allow_html=True,
                            )
                            st.code(line_text, language="python")
                    else:
                        st.caption(
                            "⚡ LLM-inferred — not a direct string literal in code "
                            "(resolved from dynamic pattern, config dict, or helper function)"
                        )
                    if desc:
                        st.caption(desc[:120] + "..." if len(desc) > 120 else desc)

        _render_tier(blocks,       "🔴", "Block — resolve before production")
        _render_tier(investigates, "🟡", "Investigate — verify with data owner")
        _render_tier(infos,        "🟢", "Informational — low risk, spot-check")
