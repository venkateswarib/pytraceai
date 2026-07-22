import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("PremiumCalc") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

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


def load_hive_table(config_key: str):
    return spark.read.table(JOB_CONFIG[config_key])


policy_df       = load_hive_table("policy_table")
endorsement_df  = load_hive_table("endorsement_table")

rate_factor_query = "(SELECT product_cd, territory_cd, base_rate, rate_multiplier FROM dbo.RateFactors WHERE active_flag = 1) rate_factors"

rate_factors_df = spark.read \
    .format("jdbc") \
    .option("url",      JDBC_CONFIG["url"]) \
    .option("dbtable",  rate_factor_query) \
    .option("driver",   JDBC_CONFIG["driver"]) \
    .option("user",     JDBC_CONFIG["user"]) \
    .option("password", JDBC_CONFIG["password"]) \
    .load()

lookup_tbl    = "territory_rate_factors"
territory_df  = spark.read.table(f"reference.{lookup_tbl}")

policy_cols = [
    "policy_id", "product_cd", "territory_cd",
    "written_premium", "policy_term_months",
    "inception_date", "expiration_date", "policy_status",
]

active_policies_df = policy_df \
    .select(*[F.col(c) for c in policy_cols]) \
    .filter(F.col("policy_status") == "ACTIVE")

endorsement_summary_df = endorsement_df \
    .groupBy("policy_id") \
    .agg(F.sum("endorsement_premium").alias("total_endorsement_premium"))

policy_with_endorsements_df = active_policies_df.join(
    endorsement_summary_df,
    on="policy_id",
    how="left",
).fillna({"total_endorsement_premium": 0.0})

rated_df = policy_with_endorsements_df.join(
    rate_factors_df,
    on=["product_cd", "territory_cd"],
    how="left",
)

final_df = rated_df.join(
    territory_df,
    on="territory_cd",
    how="left",
)

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

summary_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable(JOB_CONFIG["output_table"])

spark.stop()
