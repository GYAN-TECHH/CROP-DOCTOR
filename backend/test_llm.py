import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# --------------------------------
# Load backend/.env
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


# --------------------------------
# Get Gemini API key
# --------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found")
    print(f"Expected .env file: {ENV_PATH}")
    exit()


print("✅ GEMINI_API_KEY loaded")


# --------------------------------
# Gemini Client
# --------------------------------
try:
    client = genai.Client(api_key=api_key)
    print("✅ Gemini client connected")

except Exception as e:
    print("❌ Gemini client error:")
    print(e)
    exit()


# --------------------------------
# Test Gemini
# --------------------------------
try:

    print("\n🤖 Testing Gemini...")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=(
            "You are an agricultural assistant. "
            "Give 3 simple prevention tips for tomato late blight."
        )
    )

    print("\n" + "=" * 60)
    print("GEMINI RESPONSE")
    print("=" * 60)

    print(response.text)

    print("=" * 60)
    print("✅ GEMINI TEST SUCCESSFUL")
    print("=" * 60)

except Exception as e:

    print("\n" + "=" * 60)
    print("❌ GEMINI ERROR")
    print("=" * 60)

    print(e)

    print("=" * 60)