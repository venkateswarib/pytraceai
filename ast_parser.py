"""
PyTraceAi - ast_parser.py

Walks a PySpark script's AST and extracts lineage-relevant information:
  - sources         : tables/paths being read
  - targets         : tables/paths being written
  - joins           : join operations with left/right DataFrames, key, type
  - column_renames  : withColumnRenamed() calls
  - sql_blocks      : raw SQL strings passed to spark.sql()
"""

import ast
import textwrap
from pprint import pformat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_script(source_code: str) -> dict:
    """
    Parse a PySpark script string and return a lineage dict:
      {
        "sources":        [ "raw.claims", ... ],
        "targets":        [ "curated.claims_enriched", ... ],
        "joins":          [ {"left": "claims_df", "right": "policy_df",
                             "join_key": "policy_id == policy_id",
                             "join_type": "left"}, ... ],
        "column_renames": [ {"old_name": "claim_amount",
                             "new_name": "claim_amt_usd"}, ... ],
        "sql_blocks":     [ "SELECT ...", ... ],
      }
    """
    tree = ast.parse(source_code)
    constants = _collect_string_constants(tree)
    visitor = _LineageVisitor(constants)
    visitor.visit(tree)
    return {
        "sources":         visitor.sources,
        "targets":         visitor.targets,
        "joins":           visitor.joins,
        # AST visits outer chain calls first, so renames arrive reversed.
        "column_renames":  list(reversed(visitor.column_renames)),
        "sql_blocks":      visitor.sql_blocks,
        "opaque_logic":    _find_opaque_logic(tree, constants),
        "agg_aliases":     _find_agg_aliases(tree),
        "filters":         _find_filters(tree),
        "derived_columns": _find_derived_columns(tree),
    }


def parse_file(file_path: str) -> dict:
    """Read a PySpark script from disk and return its lineage dict."""
    with open(file_path, "r", encoding="utf-8") as fh:
        source_code = fh.read()
    return parse_script(source_code)


# ---------------------------------------------------------------------------
# Internal AST visitor
# ---------------------------------------------------------------------------

class _LineageVisitor(ast.NodeVisitor):
    """Walks every Call node in the AST and classifies PySpark operations."""

    def __init__(self, constants: dict | None = None):
        self._constants:     dict       = constants or {}
        self.sources:        list[dict] = []
        self.targets:        list[dict] = []
        self.joins:          list[dict] = []
        self.column_renames: list[dict] = []
        self.sql_blocks:     list[str]  = []

    # ---- entry point -------------------------------------------------------

    def visit_Call(self, node: ast.Call):
        """Dispatch to specialised handlers based on the call signature."""
        attr_chain = _get_attr_chain(node)

        # spark.read.table("...") / spark.read.load("...") / spark.read.csv(...)
        if self._is_read_call(attr_chain):
            self._handle_read(node, attr_chain)

        # df.write.saveAsTable("...") / df.write.save("...") / df.write.csv(...)
        elif self._is_write_call(attr_chain):
            self._handle_write(node, attr_chain)

        # df.join(other, on, how=...)
        elif attr_chain and attr_chain[-1] == "join":
            self._handle_join(node)

        # df.withColumnRenamed("old", "new")
        # Collected in reverse chain order; reversed once in parse_script.
        elif attr_chain and attr_chain[-1] == "withColumnRenamed":
            self._handle_rename(node)

        # spark.sql("SELECT ...")
        elif self._is_sql_call(attr_chain):
            self._handle_sql(node)

        self.generic_visit(node)

    # ---- read --------------------------------------------------------------

    READ_METHODS = {"table", "load", "csv", "json", "parquet", "orc",
                    "text", "jdbc"}

    def _is_read_call(self, chain: list[str]) -> bool:
        return (
            len(chain) >= 2
            and "read" in chain
            and chain[-1] in self.READ_METHODS
        )

    def _handle_read(self, node: ast.Call, chain: list[str]):
        method  = chain[-1]
        dataset = _first_string_arg(node)
        if dataset is None:
            dataset = self._resolve_arg(node)
        if dataset:
            self.sources.append({"method": method, "dataset": dataset})

    # ---- write -------------------------------------------------------------

    WRITE_METHODS = {"saveAsTable", "save", "csv", "json", "parquet",
                     "orc", "insertInto", "jdbc"}

    def _is_write_call(self, chain: list[str]) -> bool:
        return (
            len(chain) >= 2
            and "write" in chain
            and chain[-1] in self.WRITE_METHODS
        )

    def _handle_write(self, node: ast.Call, chain: list[str]):
        method  = chain[-1]
        dataset = _first_string_arg(node)
        if dataset is None:
            dataset = self._resolve_arg(node)
        if dataset:
            self.targets.append({"method": method, "dataset": dataset})

    # ---- join --------------------------------------------------------------

    def _handle_join(self, node: ast.Call):
        args = node.args
        kwargs = {kw.keyword if hasattr(kw, 'keyword') else kw.arg: kw.value
                  for kw in node.keywords}

        # left DataFrame — object the .join() is called on
        left = _receiver_name(node)

        # right DataFrame — first positional arg
        right = _name_from_arg(args[0]) if args else None

        # join condition — second positional arg (or "on" keyword)
        on_node = args[1] if len(args) > 1 else kwargs.get("on")
        join_key = _normalize_join_key(on_node) if on_node else None

        # join type — "how" keyword or third positional arg
        how_node = kwargs.get("how") if kwargs.get("how") else (
            args[2] if len(args) > 2 else None
        )
        join_type = ast.literal_eval(how_node) if how_node and isinstance(
            how_node, ast.Constant) else (
            _expr_to_str(how_node) if how_node else "inner"
        )

        self.joins.append({
            "left":      left,
            "right":     right,
            "join_key":  join_key,
            "join_type": join_type,
        })

    # ---- withColumnRenamed -------------------------------------------------

    def _handle_rename(self, node: ast.Call):
        args = node.args
        if len(args) >= 2:
            old = ast.literal_eval(args[0]) if isinstance(args[0], ast.Constant) else None
            new = ast.literal_eval(args[1]) if isinstance(args[1], ast.Constant) else None
            if old and new:
                self.column_renames.append({"old_name": old, "new_name": new})

    # ---- spark.sql() -------------------------------------------------------

    def _is_sql_call(self, chain: list[str]) -> bool:
        return len(chain) >= 2 and chain[-1] == "sql"

    def _handle_sql(self, node: ast.Call):
        sql_str = _first_string_arg(node)
        if sql_str is None:
            sql_str = self._resolve_arg(node)
        if sql_str:
            self.sql_blocks.append(textwrap.dedent(sql_str).strip())

    def _resolve_arg(self, node: ast.Call) -> str | None:
        """Resolve first arg via constant propagation: Name → string, JoinedStr → string."""
        if not node.args:
            return None
        arg = node.args[0]
        if isinstance(arg, ast.Name):
            return self._constants.get(arg.id)
        if isinstance(arg, ast.JoinedStr):
            return _try_resolve_fstring(arg, self._constants)
        return None


# ---------------------------------------------------------------------------
# Constant propagation helpers
# ---------------------------------------------------------------------------

def _collect_string_constants(tree: ast.Module) -> dict[str, str]:
    """
    Pre-pass: collect simple  name = "string literal"  assignments,
    then resolve f-strings whose every template variable is already known.
    """
    constants: dict[str, str] = {}
    # Pass 1 — plain string literals
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            constants[node.targets[0].id] = node.value.value
    # Pass 2 — f-strings where every variable is now a known constant
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.JoinedStr)):
            resolved = _try_resolve_fstring(node.value, constants)
            if resolved is not None:
                constants[node.targets[0].id] = resolved
    return constants


def _try_resolve_fstring(node: ast.JoinedStr, constants: dict[str, str]) -> str | None:
    """Return the concrete string an f-string produces, or None if any variable is unknown."""
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue):
            inner = value.value
            if isinstance(inner, ast.Name) and inner.id in constants:
                parts.append(constants[inner.id])
            else:
                return None
        else:
            return None
    return "".join(parts)


# ---------------------------------------------------------------------------
# Filter / derived-column detection
# ---------------------------------------------------------------------------

def _find_filters(tree: ast.Module) -> list[dict]:
    """
    Detect .filter(<condition>) / .where(<condition>) calls. AST can read
    the raw boolean condition directly — this is plain, visible syntax, not
    a blind spot — so it can be compared side by side against the LLM's
    business-language explanation of *why* the filter exists.
    """
    filters: list[dict] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("filter", "where") and node.args):
            continue
        condition = _expr_to_str(node.args[0])
        if condition:
            filters.append({"condition": condition, "line": node.lineno})
    return filters


def _find_derived_columns(tree: ast.Module) -> list[dict]:
    """
    Detect .withColumn("name", <expr>) calls. Like filters, this is plain
    visible syntax — AST can read both the new column name and the raw
    expression that computes it, for comparison against the LLM's semantic
    description of the derivation.
    """
    derived: list[dict] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "withColumn" and len(node.args) >= 2):
            continue
        name = _first_string_arg(node)
        expr = _expr_to_str(node.args[1])
        if name and expr:
            derived.append({"column": name, "expression": expr, "line": node.lineno})
    return derived


# ---------------------------------------------------------------------------
# Aggregation-alias detection
# ---------------------------------------------------------------------------

_AGG_FUNCS = {"sum", "avg", "count", "min", "max",
              "first", "last", "collect_list", "collect_set"}


def _find_agg_aliases(tree: ast.Module) -> list[str]:
    """
    Detect .alias("x") calls chained directly onto an aggregation function,
    e.g. F.sum("endorsement_premium").alias("total_endorsement_premium").
    This creates a brand-new column — it is not a rename of an existing one.
    An LLM sometimes still reports it as a rename (phrasing the "old name"
    either as the raw column or as the full aggregation-call text); checking
    the ast-verified new-column name here catches the hallucination
    regardless of how the LLM phrases the old side.
    """
    aliases: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "alias"):
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Call):
            rchain = _get_attr_chain(receiver)
            if rchain and rchain[-1] in _AGG_FUNCS:
                name = _first_string_arg(node)
                if name:
                    aliases.append(name)
    return aliases


# ---------------------------------------------------------------------------
# Opaque logic detection
# ---------------------------------------------------------------------------

_OPAQUE_SINKS = {"b64decode", "eval", "exec"}


def _find_decoder_functions(tree: ast.Module) -> dict[str, set[int]]:
    """
    Find functions whose body pipes one of their own parameters into
    base64.b64decode() / eval() / exec() — i.e. a helper that executes an
    opaque payload handed to it by the caller. Returns
    {function_name: {param_index, ...}}.
    """
    decoders: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        param_names = [a.arg for a in node.args.args]
        hit_indices: set[int] = set()
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func_name = (
                inner.func.attr if isinstance(inner.func, ast.Attribute) else
                inner.func.id if isinstance(inner.func, ast.Name) else None
            )
            if func_name not in _OPAQUE_SINKS:
                continue
            for arg in inner.args:
                names_in_arg = {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}
                for i, p in enumerate(param_names):
                    if p in names_in_arg:
                        hit_indices.add(i)
        if hit_indices:
            decoders[node.name] = hit_indices
    return decoders


def _find_opaque_logic(tree: ast.Module, constants: dict[str, str]) -> list[dict]:
    """
    Detect string constants that ultimately feed a base64.b64decode() /
    eval() / exec() sink — either directly, or via a helper function that
    decodes/executes one of its own parameters. AST can only see these as a
    bare Constant node; it has no way to know what the decoded value does.
    """
    const_lines: dict[str, int] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            const_lines[node.targets[0].id] = node.value.lineno

    decoders = _find_decoder_functions(tree)
    opaque: list[dict] = []
    seen: set[str] = set()

    def _record(var_name: str):
        if var_name in seen or var_name not in constants:
            return
        seen.add(var_name)
        raw = constants[var_name]
        opaque.append({
            "variable":    var_name,
            "line":        const_lines.get(var_name),
            "ast_node":    "Constant (str)",
            "raw_preview": (raw[:36] + "…") if len(raw) > 36 else raw,
        })

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Direct sink call: eval(SOME_CONSTANT) / base64.b64decode(SOME_CONSTANT)
        func_name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if func_name in _OPAQUE_SINKS:
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    _record(arg.id)
        # Call site to a known decoder helper: helper(df, SOME_CONSTANT)
        if isinstance(node.func, ast.Name) and node.func.id in decoders:
            for i in decoders[node.func.id]:
                if i < len(node.args) and isinstance(node.args[i], ast.Name):
                    _record(node.args[i].id)

    return opaque


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_attr_chain(node: ast.Call) -> list[str]:
    """
    Flatten a chained method call to a list of names, passing through
    intermediate Call nodes so multi-step chains like:
      df.write.mode("overwrite").format("parquet").saveAsTable("t")
    resolve to ["df", "write", "mode", "format", "saveAsTable"].
    """
    parts = []
    current = node.func
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Call):
            # intermediate call — step into its func to keep climbing
            current = current.func
        elif isinstance(current, ast.Name):
            parts.append(current.id)
            break
        else:
            break
    parts.reverse()
    return parts


def _first_string_arg(node: ast.Call) -> str | None:
    """Return the first positional string literal argument, or None."""
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
        node.args[0].value, str
    ):
        return node.args[0].value
    return None


def _name_from_arg(node: ast.expr) -> str | None:
    """Return the Name id if the node is a simple variable reference."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return ast.unparse(node) if node else None


def _receiver_name(node: ast.Call) -> str | None:
    """Return the name of the object a method is called on."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return ast.unparse(func.value)
    return None


def _normalize_join_key(node: ast.expr) -> str | list | None:
    """
    Reduce a join condition to a bare column name (or list of names) where
    possible, falling back to the raw unparsed expression only when necessary.

    Handles:
      df["col"] == other["col"]   ->  "col"
      df.col == other.col         ->  "col"
      "col"  (string literal)     ->  "col"
      ["col1", "col2"]            ->  ["col1", "col2"]
      F.col("col") == …           ->  "col"
      complex / unrecognised      ->  raw ast.unparse string (fallback)
    """
    if node is None:
        return None

    # Plain string: join(other, "policy_id")
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    # List of strings: join(other, ["col1", "col2"])
    if isinstance(node, ast.List):
        cols = [elt.value for elt in node.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
        return cols if cols else ast.unparse(node)

    # Comparison: df["col"] == other["col"]  /  df.col == other.col
    if (isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)):
        left_col  = _extract_col_name(node.left)
        right_col = _extract_col_name(node.comparators[0])
        if left_col and right_col:
            # same column on both sides (canonical case)
            return left_col if left_col == right_col else f"{left_col} = {right_col}"

    # F.col("col") standing alone
    if isinstance(node, ast.Call) and _get_attr_chain(node)[-1:] == ["col"]:
        val = _first_string_arg(node)
        if val:
            return val

    return ast.unparse(node)


def _extract_col_name(node: ast.expr) -> str | None:
    """Pull the bare column name out of df["col"], df.col, or F.col("col")."""
    # df["col"]  — subscript with string key
    if isinstance(node, ast.Subscript):
        s = node.slice
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            return s.value
    # df.col_name  — attribute access
    if isinstance(node, ast.Attribute):
        return node.attr
    # F.col("col") or col("col")
    if isinstance(node, ast.Call):
        return _first_string_arg(node)
    # bare string constant
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _expr_to_str(node: ast.expr) -> str | None:
    """Unparse an AST expression back to a readable string."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    import os
    from pathlib import Path

    script_path = sys.argv[1] if len(sys.argv) > 1 else \
        "sample_scripts/claims_etl.py"

    print(f"\n{'='*60}")
    print(f"  PyTraceAi — AST Parser")
    print(f"  Script: {script_path}")
    print(f"{'='*60}\n")

    result = parse_file(script_path)

    sections = [
        ("SOURCES",        result["sources"]),
        ("TARGETS",        result["targets"]),
        ("JOINS",          result["joins"]),
        ("COLUMN RENAMES", result["column_renames"]),
        ("SQL BLOCKS",     result["sql_blocks"]),
    ]

    for label, data in sections:
        print(f"--- {label} ({len(data)}) ---")
        if not data:
            print("  (none found)")
        else:
            for item in data:
                print(" ", item if isinstance(item, str) else json.dumps(item, indent=4))
        print()

    # Write JSON output to outputs/
    os.makedirs("outputs", exist_ok=True)
    stem = Path(script_path).stem
    output_path = f"outputs/{stem}_lineage.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Output saved to: {output_path}")
