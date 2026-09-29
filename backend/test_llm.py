import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit()

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        model_type="gemini-3.6-flash",
                model_version="gemini-3.5-flash",
        contents="Give 3 simple prevention tips for tomato late blight."
    )

    print("\nGemini Response:")
    print(response.text)

except Exception as e:
    print("\nGemini Error:")
    print(e)