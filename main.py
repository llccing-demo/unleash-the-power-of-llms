import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import DatetimeOutputParser, CommaSeparatedListOutputParser

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

parser_dateTime = DatetimeOutputParser()
parser_List = CommaSeparatedListOutputParser()

template = """Provide the response in format {format_instructions} 
            to the user's question {question}"""

prompt_dateTime = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_dateTime.get_format_instructions()},
)

prompt_List = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_List.get_format_instructions()},
)

response_date = model.invoke(input = prompt_dateTime.format(question="when was the first iPhone launched?"))
response_list = model.invoke(input = prompt_List.format(question="What are the four famous car brands?"))
print(response_date)
print(response_list)

