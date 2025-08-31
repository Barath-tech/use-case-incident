import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SUMMARIZER_MODEL = "openai/gpt-oss-20b:free"


def test_model():
    try:
        print("🔹 Testing OpenAI API with model:", SUMMARIZER_MODEL)
        prompt = "Hello, can you confirm the API is working?"
        response = openai.ChatCompletion.create(
            model=SUMMARIZER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        print(response)
        print("✅ API Response:", response.choices[0].message.content.strip())
    except Exception as e:
        print(f"⚠️ API Test Failed: {e}")


# Run the test
test_model()
