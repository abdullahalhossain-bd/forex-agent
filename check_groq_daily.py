import os
import requests
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

print("=" * 100)
print("GROQ DAILY QUOTA CHECK")
print("=" * 100)

for i in range(1, 15):
    key = os.getenv(f"GROQ_API_KEY_{i}")

    if not key:
        print(f"GROQ {i:02d} | MISSING")
        continue

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 1
            },
            timeout=15
        )

        print(f"\nGROQ {i:02d} | HTTP {r.status_code}")

        # Print every quota/rate-limit related header
        for name, value in r.headers.items():
            if any(x in name.lower() for x in [
                "rate", "limit", "quota", "remaining", "reset"
            ]):
                print(f"  {name}: {value}")

        if r.status_code == 200:
            print("  RESULT: DAILY QUOTA NOT EXHAUSTED (request succeeded)")

        elif r.status_code == 429:
            print("  RESULT: RATE LIMIT / QUOTA HIT")
            print("  BODY:", r.text[:1000])

        elif r.status_code == 401:
            print("  RESULT: INVALID/REVOKED KEY")

        elif r.status_code == 403:
            print("  RESULT: FORBIDDEN")

        else:
            print("  RESULT:", r.text[:500])

    except Exception as e:
        print(f"GROQ {i:02d} | ERROR | {e}")

print("\n" + "=" * 100)
