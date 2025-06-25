# splunk_mcp_server.py

from mcp.server.fastmcp import FastMCP
from splunklib.client import connect
import ollama
import os

# ------------------------------------------------------------------------------
# 🔧 MCP Initialization
# ------------------------------------------------------------------------------
mcp = FastMCP("Splunk + LLM MCP Server")

# ------------------------------------------------------------------------------
# 🔗 Splunk SDK Connection
# ------------------------------------------------------------------------------
def get_splunk_service():
    return connect(
        host=os.getenv("SPLUNK_HOST"),
        port=int(os.getenv("SPLUNK_PORT", 8089)),
        username=os.getenv("SPLUNK_USERNAME"),
        password=os.getenv("SPLUNK_PASSWORD"),
        scheme=os.getenv("SPLUNK_SCHEME", "https"),
        verify=bool(os.getenv("VERIFY_SSL", "true").lower() == "true")
    )

# ------------------------------------------------------------------------------
# 🧠 Tool: Get Splunk saved search query
# ------------------------------------------------------------------------------
@mcp.tool()
def get_use_case(name: str) -> dict:
    service = get_splunk_service()
    saved_search = service.saved_searches[name]
    current_query = saved_search.content['search']
    return {"name": name, "query": current_query}

# ------------------------------------------------------------------------------
# 🤖 Tool: Use Ollama to modify query based on description
# ------------------------------------------------------------------------------
@mcp.tool()
def generate_query_update(description: str, current_query: str) -> dict:
    prompt = f"""You are a Splunk query expert.

Current query:
{current_query}

Update the query to fulfill this requirement:
"{description}"

Only return the updated query. Do not explain anything."""
    
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    updated_query = response['message']['content'].strip()
    return {"updated_query": updated_query}

# ------------------------------------------------------------------------------
# ✏️ Tool: Update Splunk saved search
# ------------------------------------------------------------------------------
@mcp.tool()
def update_use_case_query(name: str, query: str) -> dict:
    service = get_splunk_service()
    saved_search = service.saved_searches[name]
    saved_search.update(search=query)
    saved_search.refresh()
    return {"status": "updated", "name": name, "new_query": query}

# ------------------------------------------------------------------------------
# 🚀 Run Server
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="streamable-http")


#----------------------------------------------------------------------------

import asyncio
from fastmcp import Client
from fastmcp.transports import StreamableHttpTransport

update_requests = [
    {
        "use_case": "LoginFailures",
        "description": "Add a filter for failed login attempts from internal IPs only",
        "user_email": "alice@example.com"
    },
    {
        "use_case": "FileAccessAudit",
        "description": "Limit results to last 7 days and filter for read operations",
        "user_email": "bob@example.com"
    },
]

async def process_updates():
    transport = StreamableHttpTransport(url="http://localhost:8000/mcp")
    client = Client(transport=transport)
    await client.initialize()

    for req in update_requests:
        use_case = req["use_case"]
        description = req["description"]
        user_email = req["user_email"]

        try:
            # 1. Get current query
            res = await client.call_tool("get_use_case", {"name": use_case})
            current_query = res.content["query"]

            # 2. Generate new query via LLM
            llm_res = await client.call_tool("generate_query_update", {
                "description": description,
                "current_query": current_query
            })
            updated_query = llm_res.content["updated_query"]

            # 3. Update Splunk query
            update_res = await client.call_tool("update_use_case_query", {
                "name": use_case,
                "query": updated_query
            })

            # 4. Send success email
            subject = f"[✅] Splunk Use Case Updated: {use_case}"
            body = f"Hi,\n\nThe use case '{use_case}' was successfully updated.\n\nNew query:\n{updated_query}\n\nRegards,\nAutomation System"
            send_email(user_email, subject, body)

        except Exception as e:
            # 5. Send failure email
            subject = f"[❌] Splunk Update Failed: {use_case}"
            body = f"Hi,\n\nWe encountered an error while updating use case '{use_case}':\n{str(e)}\n\nPlease investigate.\n\nRegards,\nAutomation System"
            send_email(user_email, subject, body)

if __name__ == "__main__":
    asyncio.run(process_updates())

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
import smtplib
from email.mime.text import MIMEText

def send_email(to_email: str, subject: str, body: str):
    smtp_host = "smtp.example.com"
    smtp_port = 587
    from_email = "noreply@example.com"
    password = "your_email_password"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, [to_email], msg.as_string())
        print(f"📧 Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Email failed to {to_email}: {e}")




        
update_requests = [
    {
        "use_case": "LoginFailures",
        "description": "Add a filter for failed login attempts from internal IPs only",
        "user_email": "alice@example.com"
    },
    {
        "use_case": "FileAccessAudit",
        "description": "Limit results to last 7 days and filter for read operations",
        "user_email": "bob@example.com"
    },
]

#----------------------------------------------------------------------------------

import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from openai import OpenAI
import os

# Initialize OpenAI client
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")  # e.g. https://api.openai.com/v1
)

async def update_splunk_query_with_llm(use_case: str, description: str):
    transport = StreamableHttpTransport(url="http://localhost:8000/mcp")

    async with Client(transport=transport) as client:
        # Get current query
        res = await client.call_tool("get_use_case", {"name": use_case})
        current_query = res.content["query"]

        # Call OpenAI to generate updated query
        prompt = f"""You are a Splunk SPL query expert.

Current SPL query:
{current_query}

Instruction:
{description}

Return only the updated SPL query without explanation."""

        response = llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        updated_query = response.choices[0].message.content.strip()

        print(f"🧠 Updated Query:\n{updated_query}")

        # Update in Splunk
        result = await client.call_tool("update_use_case_query", {
            "name": use_case,
            "query": updated_query
        })

        print(f"✅ Splunk updated: {result.content}")

if __name__ == "__main__":
    asyncio.run(update_splunk_query_with_llm(
        use_case="Whois - New Domain Activity",
        description="Add a filter to show only results from the last 7 days and internal IPs"
    ))
