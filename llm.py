import os
import openai
from dotenv import load_dotenv
load_dotenv()

# OpenRouter setup
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SUMMARIZER_MODEL = "gpt-oss-120b"

def gpt_summarize(text, max_tokens=50, model=SUMMARIZER_MODEL):
    print("🔹 Calling GPT for summarization...")
    prompt = f"Summarize the following incident in one short sentence:\n\n{text}"
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    print("🔹 GPT summary:", resp.choices[0].message.content.strip())
    return resp.choices[0].message.content.strip()

def gpt_category_name(text, max_tokens=10, model=SUMMARIZER_MODEL):
    print("🔹 Calling GPT for category naming...")
    prompt = (
		"You are an AI that helps categorize IT incident tickets concisely."
		"Given the following ticket(s):"
		f"{text}"
		"Respond as follows:"
		"- If you can confidently assign a category, reply with a short, specific category name (maximum 3 words, no quotes, no extra text)."
		"- If you CANNOT confidently assign a category, reply with exactly this word: Uncategorized"
		"- Do NOT explain, apologize, repeat the prompt, or add any other information."
		"- Reply with only the category name or 'Uncategorized', nothing else."
		"Examples:"
		"Correct: Network Issue"
		"Correct: Performance"
		"Correct: Uncategorized"
		"Incorrect: Sorry, I cannot categorize this."
		"Incorrect: This ticket seems to be about..."
		"Incorrect: 'Network Issue'"
		"Incorrect: Category: Network Issue"
		"Now, what is the best category for these tickets?"
	)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    raw_output = resp.choices[0].message.content.strip()

    # ✅ Clean response (avoid "Category:" or extra junk)
    clean_output = raw_output.replace("Category:", "").replace("category:", "").strip()
    clean_output = clean_output.split("\n")[0]  # only first line
    print(clean_output)
    return clean_output

