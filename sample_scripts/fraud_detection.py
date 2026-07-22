from pyspark.sql import SparkSession
import base64

spark = SparkSession.builder.appName("P&C_Loss_Ratio_Pipeline").getOrCreate()

policies_df = spark.read.table("raw.policy_premiums")
losses_df   = spark.read.table("raw.incurred_losses")

combined_df = policies_df.join(losses_df, on="policy_id", how="inner")

metadata_payload = (
    "ZGYud2l0aENvbHVtbigiTG9zc19SYXRpbyIsIChkZi5pbmN1cnJlZF9sb3NzZXMg"
    "LyBkZi5lYXJuZWRfcHJlbWl1bSkgKiAxMDAp"
)


def apply_pc_underwriting_rules(dataframe, encoded_rule):
    decoded_rule = base64.b64decode(encoded_rule).decode("utf-8")
    return eval(decoded_rule, {"df": dataframe})


final_df = apply_pc_underwriting_rules(combined_df, metadata_payload)

high_risk_df = final_df.filter(final_df.Loss_Ratio > 75.0)

high_risk_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("curated.high_risk_policies")

spark.stop()
