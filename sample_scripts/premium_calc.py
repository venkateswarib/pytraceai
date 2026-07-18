"""
premium_calc.py

P&C Premium Calculation job — computes earned premium by territory
and product line, applying rate factors from the rating engine DB
and territory multipliers from the reference data layer.

Real-world patterns that make static AST analysis incomplete:
  - Config-dict-driven table names (AST sees a variable, not a string)
  - Helper function wrapping spark.read.table (AST won't trace the call)
  - JDBC read with a dynamic query string (AST can't resolve the URL/table)
  - f-string table name with environment prefix (AST can't evaluate the expression)
  - Column list from a variable used in .select() (AST can't resolve the list)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("PremiumCalc") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ── Config dict — table names driven by environment ──────────────────────────
ENV = os.environ.get("PIPELINE_ENV", "prod")

JOB_CONFIG = {
    "policy_table":       f"{ENV}.policy_master",
    "endorsement_table":  f"{ENV}.policy_endorsements",
    "territory_table":    "reference.territory_rate_factors",
    "output_table":       f"{ENV}.premium_calculated",
}

JDBC_CONFIG = {
    "url":      os.environ.get("RATING_DB_URL", "jdbc:sqlserver://rating-engine:1433;databaseName=RatingDB"),
    "driver":   "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    "user":     os.environ.get("RATING_DB_USER", "svc_pyspark"),
    "password": os.environ.get("RATING_DB_PASS", ""),
}

# ── Pattern 1: Helper function wrapping spark.read ────────────────────────────
# AST sees load_hive_table("policy_table") — a variable, not a table name.
# It cannot trace that JOB_CONFIG["policy_table"] resolves to "prod.policy_master".

def load_hive_table(config_key: str):
    """Load a Hive table by config key."""
    return spark.read.table(JOB_CONFIG[config_key])

# ── Pattern 2: Config-dict-driven table name ──────────────────────────────────
# AST resolves spark.read.table(<something>) but <something> is a dict lookup —
# not a string literal. Source is invisible to the AST.

policy_df       = load_hive_table("policy_table")
endorsement_df  = load_hive_table("endorsement_table")

# ── Pattern 3: JDBC read with dynamic query ───────────────────────────────────
# AST can detect spark.read.jdbc() but cannot resolve the url or query —
# both are variables. The actual source table (dbo.RateFactors) is hidden.

rate_factor_query = "(SELECT product_cd, territory_cd, base_rate, rate_multiplier FROM dbo.RateFactors WHERE active_flag = 1) rate_factors"

rate_factors_df = spark.read \
    .format("jdbc") \
    .option("url",      JDBC_CONFIG["url"]) \
    .option("dbtable",  rate_factor_query) \
    .option("driver",   JDBC_CONFIG["driver"]) \
    .option("user",     JDBC_CONFIG["user"]) \
    .option("password", JDBC_CONFIG["password"]) \
    .load()

# ── Pattern 4: f-string table name ───────────────────────────────────────────
# AST sees spark.read.table(f"reference.{lookup_tbl}") — an f-string,
# not a plain string. Cannot evaluate to "reference.territory_rate_factors".

lookup_tbl    = "territory_rate_factors"
territory_df  = spark.read.table(f"reference.{lookup_tbl}")

# ── Column list from a variable ───────────────────────────────────────────────
# AST sees df.select(*policy_cols) — it cannot resolve what columns are selected.

policy_cols = [
    "policy_id", "product_cd", "territory_cd",
    "written_premium", "policy_term_months",
    "inception_date", "expiration_date", "policy_status",
]

active_policies_df = policy_df \
    .select(*[F.col(c) for c in policy_cols]) \
    .filter(F.col("policy_status") == "ACTIVE")

# ── Join endorsements to get additional premium adjustments ───────────────────
endorsement_summary_df = endorsement_df \
    .groupBy("policy_id") \
    .agg(F.sum("endorsement_premium").alias("total_endorsement_premium"))

policy_with_endorsements_df = active_policies_df.join(
    endorsement_summary_df,
    on="policy_id",
    how="left",
).fillna({"total_endorsement_premium": 0.0})

# ── Join rate factors ─────────────────────────────────────────────────────────
rated_df = policy_with_endorsements_df.join(
    rate_factors_df,
    on=["product_cd", "territory_cd"],
    how="left",
)

# ── Join territory multipliers ────────────────────────────────────────────────
final_df = rated_df.join(
    territory_df,
    on="territory_cd",
    how="left",
)

# ── Compute earned premium ────────────────────────────────────────────────────
calculated_df = final_df \
    .withColumn(
        "adjusted_premium",
        (F.col("written_premium") + F.col("total_endorsement_premium"))
        * F.col("rate_multiplier"),
    ) \
    .withColumn(
        "earned_premium",
        F.col("adjusted_premium")
        * (F.col("policy_term_months") / F.lit(12)),
    ) \
    .withColumn("calc_ts", F.current_timestamp())

# ── Aggregate by product and territory ───────────────────────────────────────
rank_window = Window.partitionBy("product_cd").orderBy(F.desc("total_earned_premium"))

summary_df = calculated_df \
    .groupBy("product_cd", "territory_cd") \
    .agg(
        F.sum("earned_premium").alias("total_earned_premium"),
        F.avg("earned_premium").alias("avg_earned_premium"),
        F.count("policy_id").alias("policy_count"),
        F.sum("adjusted_premium").alias("total_written_premium"),
    ) \
    .withColumn("territory_rank", F.rank().over(rank_window))

# ── Write output ──────────────────────────────────────────────────────────────
# Pattern: output table name also comes from config dict — AST misses this too.

summary_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable(JOB_CONFIG["output_table"])

spark.stop()
