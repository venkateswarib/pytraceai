"""
PyTraceAi - openlineage_emitter.py

Converts a merged lineage dict to an OpenLineage RunEvent.
Spec: https://openlineage.io/spec/1-0-5/OpenLineage.json
"""

import uuid
from datetime import datetime, timezone

PRODUCER = "https://github.com/pytraceai"
SCHEMA_URL = "https://openlineage.io/spec/1-0-5/OpenLineage.json"


def _infer_namespace(dataset: str) -> tuple:
    """Return (namespace, name) for a dataset string."""
    if dataset.startswith("jdbc:"):
        parts = dataset.split("/")
        return "/".join(parts[:-1]), parts[-1]
    if dataset.startswith("s3://") or dataset.startswith("s3a://"):
        without_scheme = dataset.split("://", 1)[1]
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        path = parts[1] if len(parts) > 1 else bucket
        scheme = dataset.split("://")[0]
        return f"{scheme}://{bucket}", path
    if "." in dataset:
        schema, table = dataset.split(".", 1)
        return f"hive://{schema}", table
    return "hive://default", dataset


def _column_lineage_facet(merged: dict) -> dict | None:
    """
    Build a ColumnLineageDatasetFacet: DIRECT transformations for renames
    and derived columns (value is copied/computed from named source
    columns), INDIRECT for joins and filters (they affect which rows
    survive, not a specific column's value, so they must not be reported
    as a direct copy). Source-column attribution is approximate — the
    pipeline does not track which specific upstream table each column
    came from, so all source columns are attributed to the first source
    dataset rather than guessed per-column.
    """
    sources = merged.get("sources", [])
    if not sources:
        return None
    src_ns, src_name = _infer_namespace(sources[0]["dataset"])

    fields: dict = {}
    for r in merged.get("column_renames", []):
        fields[r["new_name"]] = {
            "inputFields": [{"namespace": src_ns, "name": src_name, "field": r["old_name"]}],
            "transformations": [{"type": "DIRECT", "subtype": "IDENTITY",
                                  "description": r.get("business_reason", "column rename")}],
        }
    for t in merged.get("transformations", []):
        out_col = t.get("output_column")
        if not out_col:
            continue
        cols = t.get("source_columns") or []
        fields[out_col] = {
            "inputFields": [{"namespace": src_ns, "name": src_name, "field": c} for c in cols]
                           or [{"namespace": src_ns, "name": src_name, "field": "*"}],
            "transformations": [{"type": "DIRECT", "subtype": "TRANSFORMATION",
                                  "description": t.get("description", "")}],
        }
    if not fields:
        return None

    indirect = []
    for j in merged.get("joins", []):
        key = j.get("join_key", "")
        key_str = ", ".join(key) if isinstance(key, list) else str(key)
        indirect.append({"type": "INDIRECT", "subtype": "JOIN",
                          "description": f"{j.get('join_type','').upper()} JOIN on {key_str}"})
    for t in merged.get("transformations", []):
        if t.get("type") == "filter":
            indirect.append({"type": "INDIRECT", "subtype": "CONDITION",
                              "description": t.get("description", "")})
    if indirect:
        for f in fields.values():
            f["transformations"] = f["transformations"] + indirect

    return {"fields": fields}


def to_openlineage(merged: dict, script_name: str) -> dict:
    """Convert a merged lineage dict to an OpenLineage RunEvent dict."""
    run_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    inputs = []
    for src in merged.get("sources", []):
        ns, name = _infer_namespace(src["dataset"])
        inputs.append({
            "namespace": ns,
            "name": name,
            "facets": {
                "dataSource": {
                    "_producer": PRODUCER,
                    "_schemaURL": f"{SCHEMA_URL}#/definitions/DataSourceDatasetFacet",
                    "name": src["dataset"],
                    "uri": src["dataset"],
                },
                "pytraceai_confidence": {
                    "_producer": PRODUCER,
                    "_schemaURL": SCHEMA_URL,
                    "score": round(src.get("confidence", 1.0), 4),
                    "extractedBy": src.get("source", "both"),
                    "needsReview": src.get("needs_review", False),
                },
            },
        })

    col_lineage = _column_lineage_facet(merged)

    outputs = []
    for tgt in merged.get("targets", []):
        ns, name = _infer_namespace(tgt["dataset"])
        facets = {
            "dataSource": {
                "_producer": PRODUCER,
                "_schemaURL": f"{SCHEMA_URL}#/definitions/DataSourceDatasetFacet",
                "name": tgt["dataset"],
                "uri": tgt["dataset"],
            },
            "pytraceai_confidence": {
                "_producer": PRODUCER,
                "_schemaURL": SCHEMA_URL,
                "score": round(tgt.get("confidence", 1.0), 4),
                "extractedBy": tgt.get("source", "both"),
                "needsReview": tgt.get("needs_review", False),
            },
        }
        if col_lineage:
            facets["columnLineage"] = {
                "_producer": PRODUCER,
                "_schemaURL": f"{SCHEMA_URL}#/definitions/ColumnLineageDatasetFacet",
                **col_lineage,
            }
        outputs.append({"namespace": ns, "name": name, "facets": facets})

    return {
        "eventType": "COMPLETE",
        "eventTime": now,
        "producer": PRODUCER,
        "schemaURL": SCHEMA_URL,
        "run": {
            "runId": run_id,
            "facets": {
                "pytraceai": {
                    "_producer": PRODUCER,
                    "_schemaURL": SCHEMA_URL,
                    "overallConfidence": round(merged.get("overall_confidence", 1.0), 4),
                    "needsReviewCount": len(merged.get("needs_review", [])),
                    "joinsDetected": len(merged.get("joins", [])),
                    "businessSummary": merged.get("business_summary", ""),
                },
            },
        },
        "job": {
            "namespace": "pytraceai",
            "name": script_name,
            "facets": {
                "jobType": {
                    "_producer": PRODUCER,
                    "_schemaURL": f"{SCHEMA_URL}#/definitions/JobTypeJobFacet",
                    "processingType": "BATCH",
                    "integration": "PySpark",
                    "jobType": "JOB",
                },
                "sourceCodeLocation": {
                    "_producer": PRODUCER,
                    "_schemaURL": f"{SCHEMA_URL}#/definitions/SourceCodeLocationJobFacet",
                    "type": "local",
                    "url": f"sample_scripts/{script_name}.py",
                },
            },
        },
        "inputs":  inputs,
        "outputs": outputs,
    }
