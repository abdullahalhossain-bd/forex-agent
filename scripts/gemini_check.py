import os
from google import genai
from google.genai import errors

def check_gemini_api():
    # আপনার API Key
    api_key = "AQ.Ab8RN6Kzkhzaf-TSIO1V2DCXzcGg6WzThLMwn60dNrCwz3t_Ug" 
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("❌ ভুল: দয়া করে আপনার সঠিক Gemini API Key-টি বসান।")
        return

    print("🔄 Gemini API কানেকশন পরীক্ষা করা হচ্ছে...")
    
    try:
        # ক্লায়েন্ট ইনিশিয়ালাইজ করা হচ্ছে
        client = genai.Client(api_key=api_key)
        
        # ১. এভেইলেবল মডেলের তালিকা রিড করা
        print("🔍 আপনার অ্যাকাউন্টের জন্য এভেইলেবল মডেলগুলো খোঁজা হচ্ছে...")
        models_list = list(client.models.list())
        
        if not models_list:
            print("❌ আপনার এই API Key দিয়ে কোনো মডেল খুঁজে পাওয়া যায়নি!")
            return
            
        # সব মডেলের নাম প্রিন্ট করে দেখানো (ডিবাগিংয়ের জন্য সুবিধাজনক)
        print("\n📋 আপনার অ্যাকাউন্টে পাওয়া সমস্ত মডেলের তালিকা:")
        available_model_ids = []
        for m in models_list:
            model_id = m.name.replace("models/", "")
            print(f"  - {model_id}")
            available_model_ids.append(model_id)
        print("-" * 50)

        # ২. একের পর এক মডেল ট্রাই করা (যেটি ব্লকড নয় সেটি খুঁজে বের করতে)
        print("\n⚙️ সচল মডেল খুঁজে বের করার চেষ্টা করা হচ্ছে...")
        success = False
        
        # 'gemini-2.5-flash' কে আমরা ট্রাই করার লিস্ট থেকে বাদ দেবো, যেহেতু গুগল এটাকে ব্লক করেছে
        models_to_try = [mid for mid in available_model_ids if mid != "gemini-2.5-flash"]
        
        # যদি অন্য কোনো flash মডেল থাকে, সেগুলোকে তালিকার শুরুতে নিয়ে আসা
        models_to_try.sort(key=lambda x: "flash" in x, reverse=True)

        for model_id in models_to_try:
            print(f"🔄 {model_id} মডেলটি টেস্ট করা হচ্ছে...", end="")
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents='Hello! Respond with "OK" if you can read this.',
                )
                print(" -> ✅ সফল!")
                print(f"\n🎯 চমৎকার! আপনার জন্য সচল মডেল পাওয়া গেছে: **{model_id}**")
                print(f"🤖 মডেলের উত্তর: {response.text.strip()}")
                success = True
                break  # একটি সফল মডেল পেয়ে গেলে লুপ থেকে বের হয়ে যাবো
            except errors.APIError as e:
                # যদি নির্দিষ্ট মডেলে অ্যাক্সেস না থাকে, তাহলে এরর মেসেজটি ছোট করে প্রিন্ট করে পরের মডেলে যাবে
                print(f" -> ❌ ব্যর্থ (API Error)")
            except Exception as e:
                print(f" -> ❌ ব্যর্থ (অন্যান্য ত্রুটি)")
        
        if not success:
            print("\n❌ দুঃখিত, তালিকায় থাকা কোনো মডেল দিয়েই কানেক্ট করা সম্ভব হয়নি।")
            
    except errors.APIError as e:
        print(f"\n❌ API মূল ত্রুটি (API Error): {e}")
    except Exception as e:
        print(f"\n❌ একটি অপ্রত্যাশিত ত্রুটি ঘটেছে: {e}")

if __name__ == "__main__":
    check_gemini_api()