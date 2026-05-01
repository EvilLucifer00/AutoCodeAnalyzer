import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key from your .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Here are the models you can ACTUALLY use:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)