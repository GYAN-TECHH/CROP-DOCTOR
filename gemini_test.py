import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit()

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.7-flash",
        model_type="gemini-3.6-flash",
        model_version="gemini-3.5-flash",
        contents="Explain plant disease in one simple sentence."
    )

    print("\nGemini Response:")
    print("========================")
    print(response.text)
    print("========================")

except Exception as e:
    print("\nGemini Error:")
    print(e)