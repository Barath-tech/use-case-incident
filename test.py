import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SUMMARIZER_MODEL = "openai/gpt-oss-20b:free"

def gpt_category_name(text, max_tokens=1000, model=SUMMARIZER_MODEL):
    print("🔹 Calling GPT for category naming...")
    prompt = (
    "You are an AI that categorizes IT incident tickets into short, specific categories."
    "Given the following ticket:"
    f"\n\n{text}\n\n"
    "Respond with a category name (maximum 3 words) or 'Uncategorized' if you cannot determine a category."
    "Do not include any explanations or extra text."
)
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        print(resp)
        raw_output = resp.choices[0].message.content.strip()

        # ✅ Clean response (avoid "Category:" or extra junk)
        clean_output = raw_output.replace("Category:", "").replace("category:", "").strip()
        clean_output = clean_output.split("\n")[0]  # only first line
        print(f"Category Name: {clean_output}")
        return clean_output
    except Exception as e:
        print(f"⚠️ Error during GPT call: {e}")
        return "Uncategorized"

# Test tickets
tickets = [
    "Server is down",
    "Database connection failed",
    "High CPU usage on server",
    "Disk space is low"
]

# Test the tickets
for ticket in tickets:
    print(f"\nTicket: {ticket}")
    gpt_category_name(ticket)