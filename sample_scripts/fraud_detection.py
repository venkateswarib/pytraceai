"""
fraud_detection.py

Sample PySpark job: combines multiple sources (claims, customers,
transactions) via an embedded SQL string executed through
spark.sql(), then flags potentially fraudulent claims.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

# ---- Read multiple sources and register as temp views ----
claims_df = spark.read.table("curated.claims_enriched")
customers_df = spark.read.table("raw.customers")
transactions_df = spark.read.parquet("s3://insurance-lake/raw/transactions/")

claims_df.createOrReplaceTempView("claims")
customers_df.createOrReplaceTempView("customers")
transactions_df.createOrReplaceTempView("transactions")

# ---- Embedded SQL joining across the three sources ----
fraud_candidates_query = """
    SELECT
        c.claim_id,
        c.policy_id,
        c.claim_amt_usd,
        cu.customer_id,
        cu.risk_score,
        t.transaction_count,
        t.total_transaction_amt
    FROM claims c
    JOIN customers cu
        ON c.policy_id = cu.policy_id
    LEFT JOIN transactions t
        ON cu.customer_id = t.customer_id
    WHERE c.claim_amt_usd > 10000
      AND cu.risk_score >= 70
"""

fraud_candidates_df = spark.sql(fraud_candidates_query)

# ---- Flag high-risk claims based on business rules ----
flagged_df = fraud_candidates_df.withColumn(
    "fraud_flag",
    F.when(
        (F.col("risk_score") >= 90) | (F.col("transaction_count") > 50),
        F.lit("HIGH_RISK"),
    ).otherwise(F.lit("REVIEW")),
)

# ---- Write flagged results for investigation ----
flagged_df.write.mode("overwrite").format("delta").saveAsTable(
    "curated.fraud_flagged_claims"
)

spark.stop()
