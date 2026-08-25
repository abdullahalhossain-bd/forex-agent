import os
import requests
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

print("=" * 90)
print(f"GROQ MODEL: {model}")
print("=" * 90)

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
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply only: OK"
                    }
                ],
                "max_tokens": 3
            },
            timeout=15
        )

        h = r.headers

        remaining_requests = h.get(
            "x-ratelimit-remaining-requests", "?"
        )

        remaining_tokens = h.get(
            "x-ratelimit-remaining-tokens", "?"
        )

        reset_requests = h.get(
            "x-ratelimit-reset-requests", "?"
        )

        reset_tokens = h.get(
            "x-ratelimit-reset-tokens", "?"
        )

        if r.status_code == 200:

            print(
                f"GROQ {i:02d} | OK | "
                f"REQ_LEFT={remaining_requests} | "
                f"TOK_LEFT={remaining_tokens} | "
                f"RESET_REQ={reset_requests} | "
                f"RESET_TOK={reset_tokens}"
            )

        elif r.status_code == 429:

            print(
                f"GROQ {i:02d} | RATE_LIMIT/QUOTA | "
                f"REQ_LEFT={remaining_requests} | "
                f"TOK_LEFT={remaining_tokens}"
            )

            print(f"           {r.text[:300]}")

        elif r.status_code == 401:

            print(
                f"GROQ {i:02d} | INVALID KEY | HTTP 401"
            )

        elif r.status_code == 403:

            print(
                f"GROQ {i:02d} | FORBIDDEN | HTTP 403"
            )

        else:

            print(
                f"GROQ {i:02d} | HTTP {r.status_code} | "
                f"{r.text[:300]}"
            )

    except requests.exceptions.Timeout:

        print(
            f"GROQ {i:02d} | TIMEOUT"
        )

    except Exception as e:

        print(
            f"GROQ {i:02d} | ERROR | {e}"
        )

print("=" * 90)
