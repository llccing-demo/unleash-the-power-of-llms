import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import DatetimeOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser

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

# defining the author class for the model
class Author(BaseModel):
    number: int = Field(description="number of books written by the author")
    books: List[str] = Field(description="list of books they wrote")

user_query = "Generate the books written by 鲁迅"

# defining the output parser
output_parser = PydanticOutputParser(pydantic_object=Author)

# defining the prompt template with parser instructions
prompt1 = PromptTemplate(
    template="Answer the user query. \n{format_instructions}\n{query}\n",
    input_variables=['query'],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

# this is the latest version usecase in langchain doc
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}")
    ]
).partial(format_instructions=output_parser.get_format_instructions())

# defining the prompt
my_prompt = prompt1.format_prompt(query=user_query)

# converting the output to a string while generating the response
output = model.invoke(my_prompt.to_string())

print(output)

# Clean the output by removing code block delimiters
#cleaned_output = output.content.strip().strip('```json').strip('```')

# printing the result
print(output_parser.parse(output.content))


# Datetime and List parser
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

