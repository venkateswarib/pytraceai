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

    outputs = []
    for tgt in merged.get("targets", []):
        ns, name = _infer_namespace(tgt["dataset"])
        outputs.append({
            "namespace": ns,
            "name": name,
            "facets": {
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
            },
        })

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
