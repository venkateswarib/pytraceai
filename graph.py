"""
PyTraceAi - graph.py

Builds a directed lineage graph from the merged extractor output and
renders it to outputs/<script>_graph.png.

Layout:
  SOURCES (left)  -->  PIPELINE node (centre)  -->  TARGETS (right)

Visual encoding:
  Node color    blue = source, green = target, grey = pipeline
  Edge color    green >= 0.95, amber >= 0.70, red < 0.70
  Edge style    solid = both sources agree, dashed = one source only
  Badge "!"     node flagged needs_review
"""

import glob
import json
import os
import sys
import textwrap
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── Visual constants ──────────────────────────────────────────────────────────

NODE_COLORS = {
    "source":   "#3A86FF",
    "target":   "#2DC653",
    "pipeline": "#5C6370",
}
CONF_COLOR = {
    "high":   "#2DC653",   # >= 0.95
    "medium": "#F4A261",   # >= 0.70
    "low":    "#E63946",   # <  0.70
}
BG = "#F0F4F8"


def _conf_color(c: float) -> str:
    if c >= 0.95: return CONF_COLOR["high"]
    if c >= 0.70: return CONF_COLOR["medium"]
    return CONF_COLOR["low"]


def _edge_style(source_type: str) -> str:
    return (3, 2) if source_type in ("llm_only", "ast_only") else (None, None)


def _short_label(dataset: str, width: int = 26) -> str:
    """Trim long names (especially JDBC URLs) for node display."""
    if dataset.startswith("jdbc:"):
        tail = dataset.split("/")[-1]
        return f"JDBC:\n{tail[:width]}"
    if len(dataset) <= width:
        return dataset
    # keep schema.table but truncate schema if too long
    parts = dataset.split(".")
    if len(parts) == 2:
        schema, table = parts
        if len(dataset) > width:
            return f"{schema[:8]}...\n.{table}"
    return "..." + dataset[-(width - 3):]


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(lineage: dict, script_name: str = "Pipeline") -> nx.DiGraph:
    """
    Build a directed NetworkX DiGraph from a merged lineage dict.

    Nodes carry: node_type, label, confidence, source_type, needs_review,
                 description, method
    Edges carry: edge_type, label, confidence, source_type
    Graph carries: joins, column_renames, transformations, overall_confidence,
                   needs_review_count, script_name
    """
    G = nx.DiGraph()

    pipe_id = f"[{script_name}]"
    G.add_node(pipe_id,
               node_type="pipeline",
               label=script_name,
               confidence=lineage.get("overall_confidence", 1.0),
               source_type="both",
               needs_review=False,
               description=lineage.get("business_summary", ""))

    for src in lineage.get("sources", []):
        nid = src["dataset"]
        G.add_node(nid,
                   node_type="source",
                   label=_short_label(nid),
                   confidence=src.get("confidence", 1.0),
                   source_type=src.get("source", "both"),
                   needs_review=src.get("needs_review", False),
                   description=src.get("description", ""),
                   method=src.get("method") or "read")
        G.add_edge(nid, pipe_id,
                   edge_type="read",
                   label=src.get("method") or "read",
                   confidence=src.get("confidence", 1.0),
                   source_type=src.get("source", "both"))

    for tgt in lineage.get("targets", []):
        nid = tgt["dataset"]
        G.add_node(nid,
                   node_type="target",
                   label=_short_label(nid),
                   confidence=tgt.get("confidence", 1.0),
                   source_type=tgt.get("source", "both"),
                   needs_review=tgt.get("needs_review", False),
                   description=tgt.get("description", ""),
                   method=tgt.get("method") or "write")
        G.add_edge(pipe_id, nid,
                   edge_type="write",
                   label=tgt.get("method") or "write",
                   confidence=tgt.get("confidence", 1.0),
                   source_type=tgt.get("source", "both"))

    # Metadata stored at graph level for annotation rendering
    G.graph["joins"]             = lineage.get("joins", [])
    G.graph["column_renames"]    = lineage.get("column_renames", [])
    G.graph["transformations"]   = lineage.get("transformations", [])
    G.graph["overall_confidence"]= lineage.get("overall_confidence", 1.0)
    G.graph["needs_review_count"]= len(lineage.get("needs_review", []))
    G.graph["script_name"]       = script_name

    return G


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_graph(G: nx.DiGraph, output_path: str) -> str:
    """Render graph G to a PNG file at output_path. Returns the path."""

    sources  = [n for n, d in G.nodes(data=True) if d["node_type"] == "source"]
    targets  = [n for n, d in G.nodes(data=True) if d["node_type"] == "target"]
    pipeline = [n for n, d in G.nodes(data=True) if d["node_type"] == "pipeline"][0]

    n_src = max(len(sources), 1)
    n_tgt = max(len(targets), 1)

    # ── Manual left-centre-right layout ──────────────────────────────────────
    pos = {}
    for i, node in enumerate(sources):
        pos[node] = (0.0, 1.0 - (i + 0.5) / n_src)
    pos[pipeline] = (0.5, 0.5)
    for i, node in enumerate(targets):
        pos[node] = (1.0, 1.0 - (i + 0.5) / n_tgt)

    # ── Figure sizing ─────────────────────────────────────────────────────────
    fig_h = max(5, max(n_src, n_tgt) * 1.5 + 3.5)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.22, 1.12)
    ax.axis("off")

    script_name   = G.graph["script_name"]
    overall_conf  = G.graph["overall_confidence"]
    needs_review  = G.graph["needs_review_count"]
    conf_color    = _conf_color(overall_conf)

    ax.set_title(
        f"PyTraceAi  |  {script_name}\n"
        f"Overall Confidence: {overall_conf*100:.1f}%   "
        f"Needs Review: {needs_review}",
        fontsize=13, fontweight="bold", color="#2D3436",
        pad=14,
    )

    # ── Edges ─────────────────────────────────────────────────────────────────
    for u, v, edata in G.edges(data=True):
        conf  = edata.get("confidence", 1.0)
        stype = edata.get("source_type", "both")
        color = _conf_color(conf)
        dash, gap = _edge_style(stype)

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        props = dict(
            arrowstyle="-|>",
            color=color,
            lw=2.2 if conf >= 0.95 else 1.6,
            connectionstyle="arc3,rad=0.05",
        )
        if dash:
            props["linestyle"] = (0, (dash, gap))

        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=props, zorder=2)

        # Edge label
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        elabel = edata.get("label", "")
        ax.text(mx, my,
                f"{elabel}  {conf*100:.0f}%",
                fontsize=7, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec="none", alpha=0.85),
                zorder=3)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for node, ndata in G.nodes(data=True):
        ntype     = ndata["node_type"]
        color     = NODE_COLORS[ntype]
        x, y      = pos[node]
        label     = ndata["label"]
        conf      = ndata["confidence"]
        needs_rev = ndata["needs_review"]

        bw = 0.30 if ntype != "pipeline" else 0.24
        bh = 0.10 if ntype != "pipeline" else 0.13

        # Shadow
        shadow = mpatches.FancyBboxPatch(
            (x - bw/2 + 0.007, y - bh/2 - 0.007), bw, bh,
            boxstyle="round,pad=0.015",
            facecolor="#00000022", edgecolor="none",
            zorder=3)
        ax.add_patch(shadow)

        # Main box
        ec = "#E63946" if needs_rev else "#2D3436"
        lw = 2.5 if needs_rev else 1.5
        ls = "--" if needs_rev else "-"
        rect = mpatches.FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.015",
            facecolor=color, edgecolor=ec,
            linewidth=lw, linestyle=ls,
            alpha=0.92, zorder=4)
        ax.add_patch(rect)

        # Node label
        lines = label.split("\n")
        fs = 8.5 if ntype == "pipeline" else 7.5
        fw = "bold" if ntype == "pipeline" else "normal"
        ax.text(x, y, "\n".join(lines),
                fontsize=fs, fontweight=fw,
                color="white", ha="center", va="center",
                multialignment="center", zorder=5)

        # Confidence badge
        ax.text(x, y - bh/2 - 0.022, f"{conf*100:.0f}%",
                fontsize=7, color=_conf_color(conf),
                fontweight="bold", ha="center", va="top", zorder=5)

        # Needs-review warning marker
        if needs_rev:
            ax.text(x + bw/2 - 0.005, y + bh/2 - 0.005, "!",
                    fontsize=10, fontweight="bold",
                    color="#E63946", ha="right", va="top", zorder=6)

    # ── Column headers ────────────────────────────────────────────────────────
    ax.text(0.0,  1.08, "SOURCES",  fontsize=10, fontweight="bold",
            color=NODE_COLORS["source"],   ha="center")
    ax.text(1.0,  1.08, "TARGETS",  fontsize=10, fontweight="bold",
            color=NODE_COLORS["target"],   ha="center")
    ax.text(0.5,  1.08, "PIPELINE", fontsize=10, fontweight="bold",
            color=NODE_COLORS["pipeline"], ha="center")

    # ── Join annotations ──────────────────────────────────────────────────────
    joins = G.graph.get("joins", [])
    if joins:
        lines = []
        for j in joins:
            key = j.get("join_key", "?")
            key_str = str(key) if not isinstance(key, list) else ", ".join(key)
            lines.append(
                f"  {j.get('left','?')}  x  {j.get('right','?')}"
                f"  |  ON {key_str}  ({j.get('join_type','?').upper()})"
                f"  [{j.get('confidence',1.0)*100:.0f}%]"
            )
        ax.text(0.5, -0.12,
                "Joins:\n" + "\n".join(lines),
                fontsize=7.5, color="#4A4A4A",
                ha="center", va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.45",
                          fc="white", ec="#D0D9E6", alpha=0.95),
                zorder=5)

    # ── Transformation summary bar on pipeline node ───────────────────────────
    transforms = G.graph.get("transformations", [])
    if transforms:
        counts: dict[str, int] = {}
        for t in transforms:
            tp = t.get("type", "?")
            counts[tp] = counts.get(tp, 0) + 1
        summary = "  ".join(f"{tp}({n})" for tp, n in counts.items())
        px, py = pos[pipeline]
        ax.text(px, py - 0.085, summary,
                fontsize=6.8, color="white", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc=NODE_COLORS["pipeline"], ec="none", alpha=0.85),
                zorder=6)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=NODE_COLORS["source"],   label="Source dataset"),
        mpatches.Patch(color=NODE_COLORS["target"],   label="Target dataset"),
        mpatches.Patch(color=NODE_COLORS["pipeline"], label="PySpark pipeline"),
        Line2D([0],[0], color=CONF_COLOR["high"],   lw=2, label="Conf >= 0.95 (AST + LLM agree)"),
        Line2D([0],[0], color=CONF_COLOR["medium"], lw=2, label="Conf 0.70-0.94"),
        Line2D([0],[0], color=CONF_COLOR["low"],    lw=2, label="Conf < 0.70 (LLM only / unverified)"),
        Line2D([0],[0], color="#888", lw=1.5, linestyle=(0,(3,2)), label="Single source (needs review)"),
    ]
    ax.legend(handles=legend_items,
              loc="upper left", bbox_to_anchor=(-0.22, 1.0),
              fontsize=7.5, framealpha=0.95, edgecolor="#D0D9E6",
              title="Legend", title_fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ── Convenience loader ────────────────────────────────────────────────────────

def render_graph_from_file(merged_json_path: str) -> str:
    """Load a *_merged.json and render its graph. Returns the PNG path."""
    path    = Path(merged_json_path)
    lineage = json.loads(path.read_text(encoding="utf-8"))
    stem    = path.stem.replace("_merged", "")
    name    = stem.replace("_", " ").title()
    G       = build_graph(lineage, script_name=name)
    out     = str(path.parent / f"{stem}_graph.png")
    render_graph(G, out)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    targets = sys.argv[1:] if len(sys.argv) > 1 else \
              sorted(glob.glob("outputs/*_merged.json"))

    if not targets:
        print("No *_merged.json files found in outputs/. Run extractor.py first.")
        sys.exit(1)

    for path in targets:
        out = render_graph_from_file(path)
        print(f"  Graph saved -> {out}")
