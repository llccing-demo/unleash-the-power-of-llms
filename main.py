import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

load_dotenv()

# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
url = "https://oneapi.gptnb.me/v1"
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    openai_api_base=url,
    temperature=0.0
)

examples = [
  {
    "review": "I absolutely love this product! It exceeded my expectations.",
    "sentiment": "Positive"
  },
  {
    "review": "I'm really disappointed with the quality of this item. It didn't meet my needs.",
    "sentiment": "Negative"
  },
  {
    "review": "The product is okay, but there's room for improvement.",
    "sentiment": "Neutral"
  }
]

example_prompt = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="Review: {review}\n{sentiment}"
)

prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt=example_prompt,
    suffix="Review: {input}",
    input_variables=["input"]
)

message = prompt.format(input="The machine worked okay without much trouble.")

response = model.invoke(message)
print(response)

