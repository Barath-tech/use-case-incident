import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SUMMARIZER_MODEL = "gpt-oss-120b"

def gpt_category_name(text, max_tokens=10, model=SUMMARIZER_MODEL):
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
    "Queue MS_DWS.PORD_LGCY_C278_FAV1_BACKOUT has a Current Queue Depth of 3",
    "Job=AMTE0903150 OrderID=51hui Description=(Job that triggers th",
    "HLXP0FTE01 (ITM[1]) - File Transfer Failed - Job_Name(MQFTE_OUTBOUND_SFTP_RGISMFT_DH_G) Result_Code(",
    "MFT_MQFTE_AFG_XB60 query result is > 0.0 for 5 minutes on 'Message Age greater than 21600 secs error",
    "OPEN Custom Alert P-25025423: critical - QM - A2ILFT01 - MessageAge in Queue ITL.I0075.SALES_ORD_FRO",
    "FOD_I2699_BRADFORDWMS_WLM_FAILURE_FILEAGE_ALERT",
    "aceapp-ord009-hdx query result is > 0.0 for 5 minutes on 'KubePodNotReady in prod'",
    "B2B.I3633.DELIVERY_DATA_TO_GIST.LCQ query result is > 0.0 for 1 minutes on 'Message Age greater than"
]

# Test the tickets
for ticket in tickets:
    print(f"\nTicket: {ticket}")
    gpt_category_name(ticket)