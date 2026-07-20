import os
import requests
import time
from dotenv import load_dotenv

# .env ফাইল লোড করার জন্য
load_dotenv()

def check_groq_all_models_limits():
    # environment থেকে সব Groq keys খুঁজে বের করা
    groq_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 15) if os.getenv(f"GROQ_API_KEY_{i}")]
    
    if os.getenv("GROQ_API_KEY"):
        groq_keys.append(os.getenv("GROQ_API_KEY"))
        
    groq_keys = list(set(filter(None, groq_keys))) # Duplicate রিমুভ করা
    
    if not groq_keys:
        print("❌ ভুল: .env ফাইলে কোনো Groq API Key খুঁজে পাওয়া যায়নি!")
        return
        
    print(f"[+] Found {len(groq_keys)} Groq keys inside environment setup.")
    print("─" * 70)
    
    models_url = "https://api.groq.com/openai/v1/models"
    chat_url = "https://api.groq.com/openai/v1/chat/completions"
    
    # প্রথম সচল কি (Key) ব্যবহার করে এভেইলেবল মডেলের লিস্ট তুলে আনা
    available_models = []
    for test_key in groq_keys:
        try:
            res = requests.get(models_url, headers={"Authorization": f"Bearer {test_key}"})
            if res.status_code == 200:
                available_models = [m["id"] for m in res.json().get("data", [])]
                break
        except:
            continue
            
    # যদি কোনো কারণে ডাইনামিক লিস্ট না পাওয়া যায়, তবে স্ট্যান্ডবাই কিছু ফেমাস মডেল লিস্ট
    if not available_models:
        available_models = [
            "llama-3.3-70b-versatile", 
            "llama-3.1-8b-instant", 
            "mixtral-8x7b-32768", 
            "gemma2-9b-it"
        ]
        print("⚠️  Warning: Dynamic model list fetch failed. Using default backup list.")
    else:
        print(f"📋 Found {len(available_models)} available models from Groq API.")
        print(f"📌 Models to test: {', '.join(available_models)}")
    
    print("─" * 70)

    # প্রতিটি API Key এবং তার আন্ডারে প্রতিটি মডেল চেক করা
    for idx, key in enumerate(groq_keys, 1):
        print(f"\n🔑 [Key #{idx}] পরীক্ষা করা হচ্ছে... (Ends with: ...{key[-6:]})")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        for model in available_models:
            # রেট লিমিট বা সার্ভার স্প্যামিং এড়াতে সামান্য পজ (Pause)
            time.sleep(0.3)
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            
            print(f"  🔹 {model} -> ", end="", flush=True)
            
            try:
                response = requests.post(chat_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    print("✅ সফল! [OK]")
                    rem_req = response.headers.get("x-ratelimit-remaining-requests", "N/A")
                    rem_tok = response.headers.get("x-ratelimit-remaining-tokens", "N/n")
                    reset_req = response.headers.get("x-ratelimit-reset-requests", "N/A")
                    
                    print(f"     ↳ Requests Remaining: {rem_req} (Resets in: {reset_req})")
                    print(f"     ↳ Tokens Remaining  : {rem_tok}")
                
                elif response.status_code == 429:
                    print("❌ ব্যর্থ (API Error 429: Rate Limited)")
                    retry_after = response.headers.get("retry-after", "N/A")
                    print(f"     ↳ Retry After: {retry_after} seconds")
                    
                else:
                    print(f"❌ ব্যর্থ (Status {response.status_code})")
                    
            except Exception as e:
                print(f"❌ ত্রুটি: {str(e)[:50]}...")
                
        print("─" * 70)
        
    print("\n🏁 সমস্ত Groq Key এবং মডেলের গভীর পরীক্ষা সম্পন্ন হয়েছে।")

if __name__ == "__main__":
    check_groq_all_models_limits()