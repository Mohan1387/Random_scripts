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


#---------------------------------------

from pyspark.sql.functions import col, udf, split
import redis

# Redis Connection Settings
redis_host = "your_redis_host"
redis_port = 6379
redis_db = 0

# Connect to Redis (outside Spark)
def get_redis_connection():
    return redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

# Function to Fetch Reference Data from Redis
def fetch_from_redis(key):
    conn = get_redis_connection()
    value = conn.get(key)
    return value if value else "Unknown"

# Register as a UDF
fetch_from_redis_udf = udf(fetch_from_redis)

# Streaming Source (Delta Table)
streaming_df = spark.readStream \
    .format("delta") \
    .load("path/to/delta_source")

### MULTIPLE TRANSFORMATION FUNCTIONS ###

# Transformation 1: Clean Data (Drop Nulls)
def clean_data(df):
    return df.na.drop()

# Transformation 2: Enrich Data with Redis (Join on ID)
def enrich_data(df):
    return df.withColumn("extra_info", fetch_from_redis_udf(col("id")))

# Transformation 3: Filter Based on Some Condition
def filter_data(df):
    return df.filter(col("status") == "active")

# Transformation 4: Split a Column into Two Columns
def split_column(df, input_col="full_name"):
    return df.withColumn("first_name", split(col(input_col), " ")[0]) \
             .withColumn("last_name", split(col(input_col), " ")[1])

# Apply Transformations
transformed_df = (streaming_df
                  .transform(clean_data)
                  .transform(enrich_data)
                  .transform(filter_data)
                  .transform(lambda df: split_column(df, "full_name")))  # Assuming "full_name" exists

# Write to Delta Table
query = transformed_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "path/to/checkpoint") \
    .start("path/to/delta_output")

query.awaitTermination()

