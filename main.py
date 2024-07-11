import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
url = "https://oneapi.gptnb.me/v1"
llm = OpenAI(
    openai_api_key=api_key,
    openai_api_base=url
)

response = llm.invoke("what is the tallest building in the world?")
print(response)

