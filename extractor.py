"""
PyTraceAi - extractor.py

Dual-source lineage extraction pipeline:

  Step 1  AST pass  → parse_script() produces dict_a (structural, exact)
  Step 2  LLM pass  → Claude produces dict_b (semantic, business-aware)
  Step 3  Merge     → compare both, attach confidence scores, flag gaps
  Step 4  Enrich    → business descriptions + summary come from LLM only

Confidence scale
  1.00  both sources agree exactly
  0.85  both sources agree with minor surface differences
  0.75  AST only (structurally certain, no business context)
  0.60  LLM only (plausible but unverified by static analysis)
  0.50  partial match — values differ between sources → needs_review
"""

import json
import os
import re
import textwrap
from pathlib import Path

from openai import OpenAI

from ast_parser import parse_script

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ── LLM prompt ───────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a data engineering specialist who extracts data lineage from "
    "PySpark scripts. Return ONLY valid JSON — no prose, no markdown fences."
)

_PROMPT = """\
Analyze the PySpark script below and extract complete data lineage.

Return a JSON object with EXACTLY this structure (no extra keys):
{{
  "sources": [
    {{"dataset": "<hive table or file path>",
      "description": "<one sentence: what business data this contains>"}}
  ],
  "targets": [
    {{"dataset": "<hive table or file path>",
      "description": "<one sentence: what this output represents downstream>"}}
  ],
  "joins": [
    {{"left":        "<left DataFrame variable or table>",
      "right":       "<right DataFrame variable or table>",
      "join_key":    "<column name used to join — bare name, not expression>",
      "join_type":   "<inner | left | right | full>",
      "description": "<one sentence: business reason for this join>"}}
  ],
  "column_renames": [
    {{"old_name":        "<original column name>",
      "new_name":        "<renamed column name>",
      "business_reason": "<one sentence: why this column was renamed>"}}
  ],
  "sql_blocks": ["<raw SQL string verbatim, if any>"],
  "transformations": [
    {{"type":           "<filter | aggregate | derive | cast | window>",
      "input_df":       "<DataFrame variable name this transformation reads from>",
      "output_df":      "<DataFrame variable name this transformation produces>",
      "source_columns": ["<actual column name(s) the transformation reads — decode first if hidden behind encoding/exec/eval>"],
      "output_column":  "<column name this transformation produces (derive/cast only; omit for filter/aggregate/window)>",
      "description":    "<one sentence: what this transformation computes. If the logic is hidden behind encoding or dynamic execution (base64, exec, eval), decode it and state the EXACT formula/condition and the real source column names — never say 'applies a rule' without naming the columns and formula>"}}
  ],
  "business_summary": "<2-3 sentences: plain-English summary of the script's purpose and downstream consumers>"
}}

Script:
```python
{script}
```"""


# ── LLM caller ───────────────────────────────────────────────────────────────

def _call_llm(script: str, model: str = "anthropic/claude-sonnet-4-5") -> dict:
    """Send script to Claude via OpenRouter; return parsed lineage dict."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    client = OpenAI(base_url=_OPENROUTER_BASE, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _PROMPT.format(script=script)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
    return json.loads(raw)


# ── Merge helpers ─────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase + strip for fuzzy comparisons."""
    return (s or "").strip().lower()


def _merge_datasets(ast_list: list[dict], llm_list: list[dict]) -> list[dict]:
    """
    Compare sources or targets by dataset name.
    Both sides must have {"dataset": ...}; LLM side may add "description".
    """
    ast_map = {_norm(i["dataset"]): i for i in ast_list}
    llm_map = {_norm(i["dataset"]): i for i in llm_list}
    all_keys = sorted(set(ast_map) | set(llm_map))
    out = []
    for key in all_keys:
        in_ast = key in ast_map
        in_llm = key in llm_map
        base = ast_map.get(key, llm_map.get(key, {}))
        item = {
            "dataset":     base.get("dataset", key),
            "method":      ast_map[key].get("method") if in_ast else None,
            "description": llm_map[key].get("description", "") if in_llm else "",
        }
        if in_ast and in_llm:
            item.update(confidence=1.00, source="both",     needs_review=False)
        elif in_ast:
            item.update(confidence=0.75, source="ast_only", needs_review=False)
        else:
            item.update(confidence=0.60, source="llm_only", needs_review=True)
        out.append(item)
    return out


def _merge_joins(ast_joins: list[dict], llm_joins: list[dict]) -> list[dict]:
    """
    Match AST joins to LLM joins by join_key similarity, then score.
    AST variable names (claims_df) and LLM table names (raw.claims) differ —
    so we match on join_key + join_type, not on left/right names.
    """
    used_llm = set()
    out = []

    for a in ast_joins:
        a_key  = _norm(str(a.get("join_key", "")))
        a_type = _norm(a.get("join_type", "inner"))

        best_idx, best_score = None, 0.0
        for i, l in enumerate(llm_joins):
            if i in used_llm:
                continue
            l_key  = _norm(str(l.get("join_key", "")))
            l_type = _norm(l.get("join_type", "inner"))

            score = 0.0
            # Key match — allow substring since LLM may keep full expression
            if a_key and l_key and (a_key == l_key or a_key in l_key or l_key in a_key):
                score += 0.60
            if a_type == l_type:
                score += 0.25
            # Loose name match (variable "claims_df" vs table "raw.claims")
            a_left  = _norm(a.get("left",  ""))
            l_left  = _norm(l.get("left",  ""))
            a_right = _norm(a.get("right", ""))
            l_right = _norm(l.get("right", ""))
            if a_left and l_left and (a_left.split("_")[0] in l_left or l_left in a_left):
                score += 0.075
            if a_right and l_right and (a_right.split("_")[0] in l_right or l_right in a_right):
                score += 0.075

            if score > best_score:
                best_score, best_idx = score, i

        if best_idx is not None and best_score >= 0.60:
            used_llm.add(best_idx)
            l = llm_joins[best_idx]
            confidence = round(min(0.70 + best_score * 0.30, 1.00), 2)
            out.append({
                "left":        a.get("left"),
                "right":       a.get("right"),
                "join_key":    a.get("join_key"),     # use AST value (normalised)
                "join_type":   a.get("join_type"),
                "description": l.get("description", ""),
                "confidence":  confidence,
                "source":      "both",
                "needs_review": confidence < 0.85,
            })
        else:
            out.append({
                **a,
                "description": "",
                "confidence":  0.75,
                "source":      "ast_only",
                "needs_review": True,
            })

    # LLM-only joins — but skip if the key is already a component of a matched multi-key join
    for i, l in enumerate(llm_joins):
        if i not in used_llm:
            l_left  = _norm(l.get("left",  ""))
            l_right = _norm(l.get("right", ""))
            l_key   = _norm(str(l.get("join_key", "")))
            is_subkey = False
            for matched in out:
                if matched.get("source") not in ("both", "both_partial"):
                    continue
                m_key = matched.get("join_key")
                if not isinstance(m_key, list):
                    continue
                m_left  = _norm(matched.get("left",  ""))
                m_right = _norm(matched.get("right", ""))
                same_pair = (
                    (m_left == l_left or m_left in l_left or l_left in m_left) and
                    (m_right == l_right or m_right in l_right or l_right in m_right)
                )
                if same_pair and l_key in [_norm(k) for k in m_key]:
                    is_subkey = True
                    break
            if not is_subkey:
                out.append({
                    "left":        l.get("left"),
                    "right":       l.get("right"),
                    "join_key":    l.get("join_key"),
                    "join_type":   l.get("join_type"),
                    "description": l.get("description", ""),
                    "confidence":  0.60,
                    "source":      "llm_only",
                    "needs_review": True,
                })
    return out


def _merge_renames(ast_renames: list[dict], llm_renames: list[dict],
                    agg_aliases: list[str] | None = None) -> list[dict]:
    """
    Match renames by (old_name, new_name) pair.
    AST is structurally exact; LLM adds business_reason.
    """
    agg_alias_set = {_norm(a) for a in (agg_aliases or [])}
    llm_exact  = {(_norm(r["old_name"]), _norm(r["new_name"])): r
                  for r in llm_renames if r.get("old_name") and r.get("new_name")}
    llm_by_old = {_norm(r["old_name"]): r
                  for r in llm_renames if r.get("old_name")}
    used_llm   = set()
    out        = []

    for a in ast_renames:
        a_old = _norm(a["old_name"])
        a_new = _norm(a["new_name"])
        pair  = (a_old, a_new)

        if pair in llm_exact:
            used_llm.add(pair)
            out.append({
                "old_name":       a["old_name"],
                "new_name":       a["new_name"],
                "business_reason": llm_exact[pair].get("business_reason", ""),
                "confidence":     1.00,
                "source":         "both",
                "needs_review":   False,
            })
        elif a_old in llm_by_old:
            l = llm_by_old[a_old]
            l_pair = (_norm(l["old_name"]), _norm(l.get("new_name", "")))
            used_llm.add(l_pair)
            # AST value is authoritative; LLM agreed on old_name but differed on new
            out.append({
                "old_name":       a["old_name"],
                "new_name":       a["new_name"],
                "new_name_llm":   l.get("new_name"),    # surface the discrepancy
                "business_reason": l.get("business_reason", ""),
                "confidence":     0.80,
                "source":         "both_partial",
                "needs_review":   True,
            })
        else:
            # AST-only: rename is structurally certain, just lacks context
            out.append({
                "old_name":       a["old_name"],
                "new_name":       a["new_name"],
                "business_reason": "",
                "confidence":     0.75,
                "source":         "ast_only",
                "needs_review":   False,
            })

    # LLM-only renames — skip aggregation expressions misidentified as renames.
    # Checked two ways: the LLM sometimes phrases old_name as the raw
    # aggregation-call text (regex), and sometimes as a bare column name —
    # the ast-verified agg_aliases set catches that second case regardless
    # of phrasing, since it's keyed on the (ground-truth) new column name.
    _AGG = re.compile(r"^(sum|avg|count|min|max|first|last|collect_list|collect_set)\s*\(", re.I)
    for r in llm_renames:
        pair = (_norm(r.get("old_name", "")), _norm(r.get("new_name", "")))
        if pair not in used_llm and r.get("old_name") and r.get("new_name"):
            if _AGG.match(r["old_name"]) or _norm(r["new_name"]) in agg_alias_set:
                continue  # .agg().alias() is not a rename
            out.append({
                "old_name":       r["old_name"],
                "new_name":       r["new_name"],
                "business_reason": r.get("business_reason", ""),
                "confidence":     0.60,
                "source":         "llm_only",
                "needs_review":   True,
            })

    return out


def _overall_confidence(merged: dict) -> float:
    scores = [
        item["confidence"]
        for section in ("sources", "targets", "joins", "column_renames")
        for item in merged.get(section, [])
        if "confidence" in item
    ]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def extract_lineage(script: str, model: str = "anthropic/claude-sonnet-4-5") -> dict:
    """
    Run both passes, merge, score, and return the final lineage dict.

    Returns
    -------
    dict with keys:
      sources, targets, joins, column_renames  — merged, each item has
        confidence, source ("both"|"ast_only"|"llm_only"), needs_review
      sql_blocks          — union of both passes
      transformations     — LLM only (AST cannot derive these)
      business_summary    — LLM only
      overall_confidence  — float 0-1
      needs_review        — flat list of all items flagged needs_review=True
      _ast_raw            — raw AST dict (for debugging)
      _llm_raw            — raw LLM dict (for debugging)
    """
    print("  [1/3] Running AST parser...")
    dict_a = parse_script(script)

    print("  [2/3] Calling LLM...")
    dict_b = _call_llm(script, model=model)

    print("  [3/3] Merging and scoring...")
    merged = {
        "sources":        _merge_datasets(dict_a["sources"],        dict_b.get("sources",        [])),
        "targets":        _merge_datasets(dict_a["targets"],        dict_b.get("targets",        [])),
        "joins":          _merge_joins(   dict_a["joins"],          dict_b.get("joins",          [])),
        "column_renames": _merge_renames( dict_a["column_renames"], dict_b.get("column_renames", []),
                                           dict_a.get("agg_aliases", [])),
        # sql_blocks: union, dedup by content
        "sql_blocks":     list({s for s in (dict_a["sql_blocks"] + dict_b.get("sql_blocks", []))
                                if s and s.strip()}),
        # LLM-only enrichment — AST cannot produce these
        "transformations":  dict_b.get("transformations",  []),
        "business_summary": dict_b.get("business_summary", ""),
        # Raw outputs for transparency / debugging
        "_ast_raw": dict_a,
        "_llm_raw": dict_b,
    }
    merged["overall_confidence"] = _overall_confidence(merged)
    merged["needs_review"] = [
        {"section": sec, **item}
        for sec in ("sources", "targets", "joins", "column_renames")
        for item in merged[sec]
        if item.get("needs_review")
    ]
    return merged


def extract_lineage_from_file(
    file_path: str,
    model: str = "anthropic/claude-sonnet-4-5",
    save_output: bool = True,
) -> dict:
    """Read a PySpark script from disk, extract lineage, optionally save JSON."""
    script = Path(file_path).read_text(encoding="utf-8")
    print(f"\nPyTraceAi — extracting lineage from {file_path}")
    result = extract_lineage(script, model=model)

    if save_output:
        os.makedirs("outputs", exist_ok=True)
        stem = Path(file_path).stem

        # Output 1: AST-only
        ast_path = f"outputs/{stem}_ast.json"
        Path(ast_path).write_text(json.dumps(result["_ast_raw"], indent=2), encoding="utf-8")
        print(f"  Saved AST    -> {ast_path}")

        # Output 2: LLM-only
        llm_path = f"outputs/{stem}_llm.json"
        Path(llm_path).write_text(json.dumps(result["_llm_raw"], indent=2), encoding="utf-8")
        print(f"  Saved LLM    -> {llm_path}")

        # Output 3: Merged with confidence scores
        merged_path = f"outputs/{stem}_merged.json"
        save_dict = {k: v for k, v in result.items() if not k.startswith("_")}
        Path(merged_path).write_text(json.dumps(save_dict, indent=2), encoding="utf-8")
        print(f"  Saved Merged -> {merged_path}")

    return result


# ── CLI test harness ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Force UTF-8 output on Windows to handle LLM-generated text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    script_path = sys.argv[1] if len(sys.argv) > 1 else "sample_scripts/claims_etl.py"
    result = extract_lineage_from_file(script_path, save_output=True)

    SEP = "-" * 64

    def _badge(item: dict) -> str:
        c = item.get("confidence", 0)
        if c >= 0.95:   return "[OK ] 1.00"
        if c >= 0.80:   return f"[~  ] {c:.2f}"
        return         f"[!  ] {c:.2f}"

    print(f"\n{SEP}")
    print(f"  Overall confidence: {result['overall_confidence']:.3f}")
    print(f"  Items needing review: {len(result['needs_review'])}")
    print(SEP)

    for section, label in [
        ("sources",        "SOURCES"),
        ("targets",        "TARGETS"),
        ("joins",          "JOINS"),
        ("column_renames", "COLUMN RENAMES"),
    ]:
        items = result[section]
        print(f"\n-- {label} ({len(items)}) --")
        for item in items:
            badge = _badge(item)
            src   = item.get("source", "")
            if section in ("sources", "targets"):
                print(f"  {badge}  [{src:10s}]  {item['dataset']}")
                if item.get("description"):
                    print(f"               > {item['description']}")
            elif section == "joins":
                print(f"  {badge}  [{src:10s}]  {item['left']} JOIN {item['right']}  "
                      f"ON {item['join_key']}  ({item['join_type']})")
                if item.get("description"):
                    print(f"               > {item['description']}")
            elif section == "column_renames":
                print(f"  {badge}  [{src:10s}]  {item['old_name']}  ->  {item['new_name']}")
                if item.get("business_reason"):
                    print(f"               > {item['business_reason']}")

    print(f"\n-- TRANSFORMATIONS ({len(result['transformations'])}) --")
    for t in result["transformations"]:
        print(f"  [{t.get('type','?'):10s}]  {t.get('description','')}")

    print(f"\n-- BUSINESS SUMMARY --")
    for line in textwrap.wrap(result["business_summary"], 70):
        print(f"  {line}")

    if result["needs_review"]:
        print(f"\n-- NEEDS REVIEW ({len(result['needs_review'])}) --")
        for item in result["needs_review"]:
            sec = item.get("section", "")
            ds  = item.get("dataset") or item.get("join_key") or item.get("old_name", "?")
            print(f"  [!] [{sec}]  {ds}  (confidence={item['confidence']:.2f}, source={item['source']})")
