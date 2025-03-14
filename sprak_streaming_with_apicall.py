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
