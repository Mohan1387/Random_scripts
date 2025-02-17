import requests
from bs4 import BeautifulSoup

# List of phishing-related keywords
PHISHING_KEYWORDS = [
    "login", "signin", "password", "username", "email", "secure", "authenticate",
    "verify", "account", "banking", "credit card", "social security", "ssn",
    "pin", "security question", "reset password", "one-time password", "otp"
]

def fetch_html(url):
    """Fetch the HTML content of a given URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error for bad response
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

def extract_text(html_content):
    """Extract and return the visible text from HTML"""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").lower()  # Convert text to lowercase

def scan_for_keywords(text):
    """Scan the extracted text for phishing-related keywords"""
    detected_keywords = [word for word in PHISHING_KEYWORDS if word in text]
    return detected_keywords

def main():
    url = input("Enter the URL to scan: ").strip()
    html_content = fetch_html(url)
    
    if html_content:
        text_content = extract_text(html_content)
        detected = scan_for_keywords(text_content)
        
        if detected:
            print("\n🚨 Potential phishing indicators detected!")
            print("Keywords found:", ", ".join(detected))
        else:
            print("\n✅ No obvious phishing indicators found.")
    else:
        print("\n❌ Failed to retrieve webpage content.")

if __name__ == "__main__":
    main()
"""
response = requests.get(url, stream=True, headers=headers, timeout=10)
if "Content-Disposition" in response.headers:
    print("⚠️ Warning: The page is trying to download a file!")
    return None


from urllib.parse import urlparse

parsed_url = urlparse(url)
if parsed_url.scheme not in ["http", "https"]:
    print("🚨 Unsafe URL scheme detected! Exiting.")
    return None


import requests

def check_url_safety(url):
    API_KEY = "your_google_safe_browsing_api_key"
    google_safe_browsing_url = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    
    payload = {
        "client": {"clientId": "your_client_id", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    response = requests.post(google_safe_browsing_url, json=payload, params={"key": API_KEY})
    result = response.json()
    
    if "matches" in result:
        print("⚠️ Warning: URL is flagged as unsafe!")
        return False
    return True

url = input("Enter URL: ")
if check_url_safety(url):
    print("✅ URL is safe (not flagged).")
else:
    print("🚨 URL is potentially malicious!")

"""
