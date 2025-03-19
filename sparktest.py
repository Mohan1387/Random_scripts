from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, broadcast

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DeltaStreamEnrichment") \
    .config("spark.sql.streaming.schemaInference", "true") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Read streaming data from Delta Table
stream_df = spark.readStream \
    .format("delta") \
    .option("ignoreChanges", "true") \
    .table("your_source_table")

# Extract unique domains for lookup
domain_lookup_df = stream_df.select("domains").distinct()

# Collect unique domains as a lookup list (Batch Query)
domain_list = domain_lookup_df.agg(collect_set(col("domains"))).collect()[0][0]

# Create a DataFrame from the lookup list
lookup_df = spark.createDataFrame([(d,) for d in domain_list], ["domain"])

# Broadcast lookup table
lookup_df = broadcast(lookup_df)

# Enrich the main stream
enriched_df = stream_df.join(lookup_df, stream_df.domains == lookup_df.domain, "left")

# Write the enriched stream to the console
query = enriched_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
