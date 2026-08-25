import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY_1")

if not key:
    print("GROQ_API_KEY_1 not found")
    raise SystemExit

headers = {
    "Authorization": f"Bearer {key}"
}

urls = [
    "https://api.groq.com/openai/v1/usage",
    "https://api.groq.com/openai/v1/usage/aggregated",
    "https://api.groq.com/openai/v1/usage/requests",
]

for url in urls:
    print("\n" + "=" * 90)
    print("URL:", url)
    print("=" * 90)

    try:
        r = requests.get(url, headers=headers, timeout=15)

        print("HTTP:", r.status_code)
        print("Response:")
        print(r.text[:3000])

    except Exception as e:
        print("ERROR:", e)
