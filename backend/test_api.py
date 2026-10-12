import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found")
    exit()

print("✅ HF_TOKEN loaded")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Give one simple treatment tip for tomato late blight."
        }
    ]
)

print("\n🤖 Hugging Face Response:")
print(response.choices[0].message.content)