import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain 

load_dotenv()
#########   LLM Chain     ###############
# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
url = "https://oneapi.gptnb.me/v1"
model = ChatOpenAI(
    #model="claude-3-sonnet-20240229",
    model="gpt-4o",
    #model="gemini-1.5-pro-latest",
    openai_api_key=api_key,
    openai_api_base=url,
)

prompt = ChatPromptTemplate.from_messages(
    [("user", "Name the author of the book {book}")],
)

# chain = prompt | model | StrOutputParser()

# resp = chain.invoke({"book": "The Da Vinci Code"})

# print(resp)

#########   LLM Chain     ###############

chain_1 = LLMChain(llm=model, prompt=prompt)
prompt_2 = ChatPromptTemplate.from_messages(
    [("user", "Write a 50-word biography fro the following author:{author_name}")]
)
chain_2 = LLMChain(llm=model, prompt=prompt_2)

prompt_3 = ChatPromptTemplate.from_messages(
    [("user", "translate all contents {contents} to Chinese")]
)
chain_3 = LLMChain(llm=model, prompt=prompt_3)

#chain_1 = prompt | model | StrOutputParser()
#chain_2 = prompt_2 | model | StrOutputParser()
#chain_3 = prompt_3 | model | StrOutputParser()


simple_sequential_chain = SimpleSequentialChain(chains=[chain_1, chain_2, chain_3], verbose=True)
#simple_sequential_chain = RunnableSequence([chain_1, chain_2, chain_3], verbose=True)

resp = simple_sequential_chain.invoke("The Da Vinci Code")
print(resp)
