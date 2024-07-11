import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
url = "https://oneapi.gptnb.me/v1"
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    openai_api_base=url
)

messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="Hi !")
]

response = model.invoke(messages)
print(response)

