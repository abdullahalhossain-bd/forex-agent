import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, GoogleAPIError

# .env ফাইল লোড করা হচ্ছে
load_dotenv()

# আপনার .env ফাইলে থাকা API Keys (কমা দিয়ে আলাদা করা থাকলে বা আলাদা ভেরিয়েবল থাকলে)
# এখানে উদাহরণ হিসেবে ৩/৪টি কী চেক করার লজিক রাখা হলো
api_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4")
]
# None বা খালি কীগুলো ফিল্টার করা
api_keys = [key for key in api_keys if key]

print(f"🔄 .env থেকে মোট {len(api_keys)}টি API Key পাওয়া গেছে।")
preferred_model = os.getenv("MODEL_NAME", "gemini-flash-lite-latest")
print(f"⚙️ প্রছন্দনীয় মডেল (.env): {preferred_model}\n")

# ফরেক্স এবং জেনারেল কাজের জন্য মডেলের র‍্যাংকিং (Priority Order)
MODEL_PRIORITY = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-robotics-er-1.6-preview"
]

successful_models_global = set()

for idx, key in enumerate(api_keys, 1):
    print("=" * 70)
    # সিকিউরিটির জন্য কী-এর কিছু অংশ মাস্ক করে দেখানো
    masked_key = f"{key[:9]}...{key[-4:]}" if len(key) > 15 else "Invalid Format"
    print(f"🔑 [Key #{idx}] পরীক্ষা করা হচ্ছে... (Key: {masked_key})")
    
    if not key.startswith("AIza"):
        print("   ⚠️  সতর্কতা: এই Key-টির ফরম্যাট সঠিক নয় (গুগল কি সাধারণত AIza দিয়ে শুরু হয়)।")
        
    try:
        genai.configure(api_key=key)
        
        print("   🔍 অ্যাকাউন্টের উপলব্ধ সমস্ত মডেল খোঁজা হচ্ছে...")
        available_models = [m.name.split('/')[-1] for m in genai.list_models()]
        print(f"   📋 মোট {len(available_models)}টি মডেল পাওয়া গেছে।")
        print("   ⚙️ প্রতিটি মডেল আলাদাভাবে রেসপন্স টেস্ট করা হচ্ছে:")
        
        # শুধুমাত্র আমাদের প্রায়োরিটি লিস্ট বা টেস্ট করার মতো মডেল টেস্ট করা
        # সব ৫৪টি মডেল টেস্ট করলে কোটা দ্রুত শেষ হয়, তাই সচলগুলো চেক করছি
        models_to_test = [m for m in available_models if any(p in m for p in MODEL_PRIORITY)]
        
        # যদি লিস্ট খালি থাকে, তবে ডিফল্ট কিছু মডেল টেস্ট করবে
        if not models_to_test:
            models_to_test = MODEL_PRIORITY
            
        for model_name in models_to_test:
            try:
                # রেট লিমিট এড়াতে সামান্য বিরতি
                time.sleep(0.5)
                
                model = genai.GenerativeModel(model_name)
                # একটি ছোট এবং দ্রুত টেস্ট রিকোয়েস্ট
                response = model.generate_content("Hi", generation_config={"max_output_tokens": 5})
                
                if response.text:
                    print(f"      🔹 {model_name} -> ✅ সফল! [OK]")
                    successful_models_global.add(model_name)
            except ResourceExhausted:
                print(f"      🔹 {model_name} -> ❌ ব্যর্থ (API Error 429): Quota Exceeded")
            except InvalidArgument:
                print(f"      🔹 {model_name} -> ❌ ব্যর্থ (API Error 400): Invalid Argument")
            except Exception as e:
                err_msg = str(e)[:50]
                print(f"      🔹 {model_name} -> ❌ ব্যর্থ: {err_msg}...")
                
        print(f"\n   🎯 চমৎকার! Key #{idx} আংশিক বা সম্পূর্ণ সচল।")
        
    except GoogleAPIError as auth_err:
        print("   -> ❌ ক্লায়েন্ট লেভেলে ব্যর্থ!")
        print(f"   🔴 API Error: Auth বা ক্রেডেনশিয়াল সঠিক নয়। ({str(auth_err)[:100]})")
    except Exception as e:
        print(f"   -> ❌ সাধারণ ত্রুটি: {str(e)[:100]}")

print("=" * 70)
print("🏁 সমস্ত মডেলের গভীর পরীক্ষা সম্পন্ন হয়েছে।\n")

# Forex এর জন্য বেস্ট মডেল সিলেকশন রেজাল্ট
print("📊 [Forex Project Recommendation Results]")
if successful_models_global:
    best_chosen = None
    for target in MODEL_PRIORITY:
        if target in successful_models_global:
            best_chosen = target
            break
            
    if best_chosen:
        print(f"🏆 আপনার উপলব্ধ সচল মডেলগুলোর মধ্যে Forex-এর জন্য বেস্ট মডেল: 🔥 {best_chosen} 🔥")
        print(f"💡 আপনার .env ফাইলে এটি আপডেট করুন: MODEL_NAME={best_chosen}")
    else:
        # যদি প্রায়োরিটি লিস্টের বাইরের কোনো মডেল সচল হয়
        any_model = list(successful_models_global)[0]
        print(f"✅ সচল মডেল পাওয়া গেছে: {any_model} (এটি আপনার ফরেক্স প্রজেক্টে ব্যবহার করতে পারেন)।")
else:
    print("❌ দুঃখিত, কোনো মডেলই সফলভাবে রেসপন্স করেনি। অনুগ্রহ করে আপনার API Key বা কোটা চেক করুন।")
print("=" * 70)