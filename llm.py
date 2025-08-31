import os
import re
import time
import openai
from dotenv import load_dotenv
from openai.error import RateLimitError 

load_dotenv()

# OpenRouter setup
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SUMMARIZER_MODEL = "openai/gpt-oss-20b:free"


# def gpt_summarize(text, model=SUMMARIZER_MODEL):
#     print("🔹 Calling GPT for summarization...")
#     prompt = f"Summarize the following incident in one short sentence:\n\n{text}"
#     resp = openai.ChatCompletion.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#     )
#     summary = resp.choices[0].message.content.strip()
#     print(f"✅ Summary: {summary}")
#     time.sleep(2)  # prevent hitting rate-limit
#     return summary


def gpt_category_name(text, model=SUMMARIZER_MODEL):
    print("🔹 Calling GPT for category naming...")
    prompt = (
        "You are an AI that categorizes IT incident tickets.\n"
        f"Ticket:\n{text}\n\n"
        "Rules:\n"
        "- Reply ONLY with a short category name (≤3 words) OR 'Uncategorized'.\n"
        "- No reasoning, no explanations, no quotes.\n"
        "- Examples:\nNetwork Issue\nPerformance\nUncategorized"
    )

    while True:  # Retry loop
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if "assistantfinal" in raw:
                raw = raw.split("assistantfinal")[-1].strip()

            # ✅ Extract clean category
            clean = re.sub(r"[^A-Za-z0-9 ]+", "", raw).strip()
            clean = clean.split("\n")[0]
            print(f"✅ Category: {clean}")
            time.sleep(2)  # Prevent hitting rate limits
            return clean
        except RateLimitError:
            print("⚠️ Rate limit reached. Retrying after 5 seconds...")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            print(f"⚠️ Error during GPT call: {e}")
            return "Uncategorized"
