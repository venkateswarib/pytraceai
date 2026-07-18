"""
claims_etl.py

P&C Insurance ETL: joins raw claims with policy master on policy_id,
applies column renames for downstream consumption, and writes the
enriched result to the curated layer.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

spark = SparkSession.builder \
    .appName("ClaimsETL") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ---- Read sources ----
claims_df = spark.read.table("raw.claims")
policy_df = spark.read.table("raw.policy")

# ---- Join claims to policy on policy_id (left join - keep all claims) ----
joined_df = claims_df.join(
    policy_df,
    claims_df["policy_id"] == policy_df["policy_id"],
    how="left",
)

# ---- Select and rename columns for downstream consumers ----
enriched_df = joined_df.select(
    claims_df["claim_id"],
    claims_df["policy_id"],
    claims_df["claim_amount"],
    claims_df["claim_date"],
    claims_df["loss_type"],
    claims_df["claimant_first_nm"],
    claims_df["claimant_last_nm"],
    policy_df["policy_holder_name"],
    policy_df["policy_type"],
    policy_df["coverage_limit"],
    policy_df["deductible_amount"],
    policy_df["effective_date"],
    policy_df["expiration_date"],
    policy_df["underwriting_region"],
)

# Rename raw column names to enterprise standard names
renamed_df = enriched_df \
    .withColumnRenamed("claim_amount",      "claim_amt_usd") \
    .withColumnRenamed("claim_date",        "claim_dt") \
    .withColumnRenamed("policy_holder_name","customer_full_name") \
    .withColumnRenamed("policy_type",       "product_line_cd") \
    .withColumnRenamed("coverage_limit",    "max_coverage_amt_usd") \
    .withColumnRenamed("deductible_amount", "deductible_amt_usd") \
    .withColumnRenamed("effective_date",    "policy_effective_dt") \
    .withColumnRenamed("expiration_date",   "policy_expiration_dt") \
    .withColumnRenamed("underwriting_region","uw_region_cd")

# ---- Derive additional business fields ----
final_df = renamed_df \
    .withColumn("claim_amt_usd",     F.col("claim_amt_usd").cast(DecimalType(15, 2))) \
    .withColumn("max_coverage_amt_usd", F.col("max_coverage_amt_usd").cast(DecimalType(15, 2))) \
    .withColumn("deductible_amt_usd", F.col("deductible_amt_usd").cast(DecimalType(15, 2))) \
    .withColumn("net_payable_amt",
                F.greatest(F.col("claim_amt_usd") - F.col("deductible_amt_usd"), F.lit(0))) \
    .withColumn("is_catastrophic",
                F.when(F.col("claim_amt_usd") > 500000, F.lit(True)).otherwise(F.lit(False))) \
    .withColumn("etl_load_ts",       F.current_timestamp()) \
    .filter(F.col("claim_amt_usd").isNotNull())

# ---- Write curated output ----
final_df.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("curated.claims_enriched")

spark.stop()
