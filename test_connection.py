from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the .env file so GOOGLE_API_KEY becomes available
load_dotenv()

# Create a connection to Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Send one message and get a reply
response = llm.invoke("Say hello in one short sentence.")

print(response.text)