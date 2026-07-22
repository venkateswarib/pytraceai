"""
fraud_detection.py — P&C Loss Ratio Pipeline

Demonstrates AST blind spot: the core underwriting formula is base64-encoded
to protect proprietary business logic at rest in source control.

  AST finds:   2 sources (raw.policy_premiums, raw.incurred_losses),
               1 join (policy_id, inner), 1 target (curated.high_risk_policies)
  AST misses:  the Loss_Ratio derived column — hidden inside the encoded payload
  LLM finds:   everything + decodes the formula:
               Loss_Ratio = (incurred_losses / earned_premium) × 100

Static analysis sees the data flow skeleton (which tables, which join).
The LLM decodes the encoded payload and fills in the business transformation
that makes the lineage meaningful for data governance.
"""

from pyspark.sql import SparkSession
import base64

spark = SparkSession.builder.appName("P&C_Loss_Ratio_Pipeline").getOrCreate()


# ── Source reads — fully visible to AST ──────────────────────────────────────
# AST can resolve these literal table names and record them as sources.
policies_df = spark.read.table("raw.policy_premiums")
losses_df   = spark.read.table("raw.incurred_losses")


# ── Join — fully visible to AST ───────────────────────────────────────────────
# AST can detect .join() calls and extract the key and join type.
combined_df = policies_df.join(losses_df, on="policy_id", how="inner")


# ── AST BLIND SPOT: underwriting formula is base64-encoded ───────────────────
# The loss-ratio formula is encoded to protect proprietary underwriting IP
# from being readable in source control. A standard static AST parser sees
# a plain string literal assigned to a variable — it cannot decode or execute
# it, and therefore has no way to discover that a "Loss_Ratio" derived column
# is being created from incurred_losses and earned_premium.
#
# The LLM recognises the base64 pattern, decodes the payload, and recovers:
#   → df.withColumn("Loss_Ratio", (df.incurred_losses / df.earned_premium) * 100)

metadata_payload = (
    "ZGYud2l0aENvbHVtbigiTG9zc19SYXRpbyIsIChkZi5pbmN1cnJlZF9sb3NzZXMg"
    "LyBkZi5lYXJuZWRfcHJlbWl1bSkgKiAxMDAp"
)


def apply_pc_underwriting_rules(dataframe, encoded_rule):
    """Executes business rules injected dynamically at runtime."""
    decoded_rule = base64.b64decode(encoded_rule).decode("utf-8")
    return eval(decoded_rule, {"df": dataframe})


# Execute the hidden logic — AST sees a function call but cannot follow inside
final_df = apply_pc_underwriting_rules(combined_df, metadata_payload)


# ── Filter and write — visible to AST ────────────────────────────────────────
# AST can see the filter and the write target. It knows *that* a filter on
# Loss_Ratio exists (from the attribute reference), but not *what* Loss_Ratio
# represents because the formula that creates it is encoded above.
high_risk_df = final_df.filter(final_df.Loss_Ratio > 75.0)

high_risk_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("curated.high_risk_policies")

spark.stop()
