# pip install fastapi aioredis asyncio whois-api uvicorn gunicorn

import aioredis
import asyncio
from fastapi import FastAPI, HTTPException
from whoisapi import Client

# Initialize FastAPI
app = FastAPI()

# Whois API client setup
WHOIS_API_KEY = "your_api_key"
whois_client = Client(api_key=WHOIS_API_KEY)

# Redis connection
redis_client = None

@app.on_event("startup")
async def startup():
    """Initialize Redis connection on FastAPI startup."""
    global redis_client
    redis_client = await aioredis.from_url("redis://localhost", decode_responses=True)

async def fetch_whois_data(domain: str):
    """Fetch Whois data from the external API using whois-api package."""
    try:
        response = whois_client.get(domain)
        return response if response else {"error": f"No data for {domain}"}
    except Exception as e:
        return {"error": f"Failed to fetch {domain}", "message": str(e)}

async def process_domain(domain: str):
    """Check Redis first, fetch from Whois API if not cached, then store in Redis permanently."""
    cached_data = await redis_client.get(domain)
    if cached_data:
        return cached_data  # Return cached Whois data

    # Fetch Whois data from API
    whois_data = await fetch_whois_data(domain)

    # Store in Redis permanently (no expiration)
    await redis_client.set(domain, str(whois_data))

    return whois_data

async def process_domains_in_batches(domains):
    """Process domains in batches of 50 requests per second."""
    results = []
    batch_size = 50  # Whois API allows 50 requests per second

    for i in range(0, len(domains), batch_size):
        batch = domains[i : i + batch_size]  # Create a batch of 50
        tasks = [process_domain(domain) for domain in batch]

        # Execute requests concurrently with asyncio
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Rate limit: wait 1 second before the next batch
        await asyncio.sleep(1)

    return results

@app.post("/whois/bulk")
async def bulk_whois_lookup(domains: list[str]):
    """Handles bulk Whois lookup requests (1000 domains per request)."""
    if len(domains) > 1000:
        raise HTTPException(status_code=400, detail="Limit is 1000 domains per request")

    results = await process_domains_in_batches(domains)
    return {"results": results}


"""
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 script_name:app

redis-cli info memory


Check Cached Whois Data

redis-cli keys '*'
redis-cli get "example.com"

Test Bulk Whois Lookup
Run a test request with 1000 domains:

curl -X POST "http://127.0.0.1:8000/whois/bulk" -H "Content-Type: application/json" -d '["example1.com", "example2.com", ..., "example1000.com"]'


#######################
# Redis Memory Config #
#######################

# Limit Redis to 100GB of RAM usage
maxmemory 100gb
# Evict least recently used (LRU) keys only if needed
maxmemory-policy volatile-lru

#######################
# Persistence Config  #
#######################

# Append-only file (AOF) mode: ensures data is saved to disk
appendonly yes
appendfilename "appendonly.aof"

# Optimize AOF for SSD performance
auto-aof-rewrite-percentage 50
auto-aof-rewrite-min-size 64mb

# Enable periodic snapshots (for quick reloads)
save 300 1  # Save every 5 mins if at least 1 key is modified

# Store Redis database files on SSD
dir /var/lib/redis

sudo systemctl restart redis

"""

import aioredis
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from whoisapi import Client
import time

# Initialize FastAPI
app = FastAPI()

# Whois API client setup
WHOIS_API_KEY = "your_api_key"
whois_client = Client(api_key=WHOIS_API_KEY)

# Redis connection
redis_client = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("whois_api.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

@app.on_event("startup")
async def startup():
    """Initialize Redis connection on FastAPI startup."""
    global redis_client
    redis_client = await aioredis.from_url("redis://localhost", decode_responses=True)

async def fetch_whois_data(domain: str):
    """Fetch Whois data from the external API using whois-api package."""
    try:
        start_time = time.time()  # Start timing request
        response = whois_client.get(domain)
        end_time = time.time()  # End timing request

        elapsed_time = round(end_time - start_time, 3)
        logging.info(f"Whois API Request | Domain: {domain} | Time Taken: {elapsed_time}s")

        return response if response else {"error": f"No data for {domain}"}
    except Exception as e:
        logging.error(f"Whois API Failure | Domain: {domain} | Error: {str(e)}")
        return {"error": f"Failed to fetch {domain}", "message": str(e)}

async def process_domain(domain: str):
    """Check Redis first, fetch from Whois API if not cached, then store in Redis permanently."""
    cached_data = await redis_client.get(domain)
    if cached_data:
        logging.info(f"Cache Hit | Domain: {domain}")
        return cached_data  # Return cached Whois data

    # Fetch Whois data from API
    whois_data = await fetch_whois_data(domain)

    # Store in Redis permanently (no expiration)
    await redis_client.set(domain, str(whois_data))
    logging.info(f"Cache Store | Domain: {domain}")

    return whois_data

async def process_domains_in_batches(domains):
    """Process domains in batches of 50 requests per second."""
    results = []
    batch_size = 50  # Whois API allows 50 requests per second

    for i in range(0, len(domains), batch_size):
        batch = domains[i : i + batch_size]  # Create a batch of 50
        tasks = [process_domain(domain) for domain in batch]

        # Execute requests concurrently with asyncio
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Rate limit: wait 1 second before the next batch
        await asyncio.sleep(1)

    return results

@app.post("/whois/bulk")
async def bulk_whois_lookup(domains: list[str]):
    """Handles bulk Whois lookup requests (1000 domains per request)."""
    if len(domains) > 1000:
        raise HTTPException(status_code=400, detail="Limit is 1000 domains per request")

    logging.info(f"Bulk Whois Request | Domains: {len(domains)}")
    results = await process_domains_in_batches(domains)
    return {"results": results}


from whoisapi import Client
import logging
import time
import asyncio

WHOIS_API_KEY = "your_api_key"
whois_client = Client(api_key=WHOIS_API_KEY)

async def fetch_whois_data(domain: str, timeout: int = 30):
    """Fetch Whois data asynchronously with increased timeout."""
    try:
        start_time = time.time()  # Start timing request

        # Run Whois lookup in a thread executor with a timeout
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, whois_client.rawdata, domain),
            timeout=timeout
        )

        end_time = time.time()  # End timing request
        elapsed_time = round(end_time - start_time, 3)

        logging.info(f"Whois API Request | Domain: {domain} | Time Taken: {elapsed_time}s")

        return response if response else {"error": f"No data for {domain}"}

    except asyncio.TimeoutError:
        logging.error(f"Whois API Timeout | Domain: {domain} | Timeout: {timeout}s")
        return {"error": f"Timeout while fetching {domain}", "timeout": timeout}

    except Exception as e:
        logging.error(f"Whois API Failure | Domain: {domain} | Error: {str(e)}")
        return {"error": f"Failed to fetch {domain}", "message": str(e)}
