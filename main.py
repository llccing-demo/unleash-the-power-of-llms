import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
url = "https://oneapi.gptnb.me/v1"
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    openai_api_base=url
)

email_template = ChatPromptTemplate.from_template(
    "Create an invitation email to the recipinet that is {recipient_name} \
 for an event that is {event_type} in a language that is {language} \
 Mention the event location that is {event_location} \
 and event date that is {event_date}. \
 Also write few sentences about the event description that is {event_description} \
 in style that is {style} "
)

message = email_template.format(
    style = "enthusiastic tone",
    language = "American english",
    recipient_name = "Rowan",
    event_type="product launch",
    event_date="July 11, 2024",
    event_location="Dalian, China",
    event_description="an exciting unveiling of latest innovations"
)

response = model.invoke(message)
print(response)

