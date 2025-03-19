import time
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import pandas_udf

# Initialize Spark Session
spark = SparkSession.builder.appName("WhoisAPILookup").getOrCreate()

# API Configuration
WHOIS_API_URL = "http://your-api-server.com/whois/"
RATE_LIMIT = 50  # API allows only 50 requests per second

# Function to Call Whois API in Batches
def call_whois_api(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calls the external Whois API in batches, respecting the rate limit.
    """
    results = []
    batch_size = len(batch_df)
    domains = batch_df["domain"].tolist()

    for i in range(0, batch_size, RATE_LIMIT):  # Process in chunks of 50
        batch = domains[i:i + RATE_LIMIT]  # Take 50 at a time
        batch_results = []

        for domain in batch:
            try:
                response = requests.get(f"{WHOIS_API_URL}{domain}")

                if response.status_code == 200:
                    batch_results.append(response.json())  # Store API response
                else:
                    batch_results.append({"error": "API_ERROR"})
            
            except Exception as e:
                batch_results.append({"error": str(e)})

            time.sleep(1 / RATE_LIMIT)  # Ensure we do not exceed 50 RPS

        results.extend(batch_results)

    batch_df["api_response"] = results
    return batch_df

# Register as Pandas UDF
@pandas_udf(StringType())
def call_whois_api_udf(domain_col: pd.Series) -> pd.Series:
    return call_whois_api(domain_col.to_frame())["api_response"]

# Define Schema for Input Data
schema = StructType([StructField("domain", StringType(), True)])

# Read Streaming Data
df = spark.readStream.format("csv").schema(schema).option("path", "/path/to/data").load()

# Apply Whois API Lookup with Rate-Limiting
df_with_api = df.withColumn("api_response", call_whois_api_udf(col("domain")))

# Write to Console or Storage
df_with_api.writeStream.format("console").start().awaitTermination()


#---------------------------------------------------------------

import requests
import pandas as pd
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark Session
spark = SparkSession.builder.appName("WHOISStreaming").getOrCreate()

# WHOIS API Configuration
API_URL = "https://your-whois-api.com/query"  # Replace with actual API
HEADERS = {
    "Content-Type": "application/json",
    "API-Key": "your_api_key"  # Replace with actual API key
}

def whois_api_lookup_batch(domains):
    """Call WHOIS API with a batch of domains and return responses."""
    if not isinstance(domains, list):  # Ensure input is a list
        return ["Invalid Input"]

    try:
        payload = json.dumps({"domains": domains.tolist()})  # Convert Pandas Series to list
        response = requests.post(API_URL, data=payload, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("results", ["Error"] * len(domains))
        else:
            return ["Error"] * len(domains)  # Handle failures gracefully
    except Exception as e:
        return [str(e)] * len(domains)  # Handle exceptions

# Define Pandas UDF
@pandas_udf(ArrayType(StringType()))
def whois_udf(domain_series: pd.Series) -> pd.Series:
    return whois_api_lookup_batch(domain_series)

# Read Streaming Data (Example: Assume Kafka Source)
streaming_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")  # Replace with actual Kafka details
    .option("subscribe", "whois_topic")  # Replace with actual topic
    .load()
    .selectExpr("CAST(value AS STRING) as domain")  # Convert Kafka message to column
)

# Apply WHOIS API lookup on streaming data
streaming_df = streaming_df.withColumn("whois_info", whois_udf(streaming_df["domain"]))

# Write Output Stream (Example: Console Output)
query = (
    streaming_df.writeStream
    .outputMode("append")  # Change as per use case
    .format("console")  # Can use "parquet", "kafka", "delta", etc.
    .start()
)

query.awaitTermination()

