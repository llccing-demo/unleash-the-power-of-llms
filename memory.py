import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

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

memory = ConversationBufferMemory()
memory.save_context(
  { "input": "Alex is a 9-year old boy."},
  { "output": "Hello Alex! How can I assist you today?"}
)

memory.save_context(
  { "input": "Alex likes to play football"},
  { "output": "That's great to hear!" }
)

conversation = ConversationChain(
  llm = model,
  memory = memory,
  verbose = True
)

resp = conversation.predict(input="how old is Alex?")
print(resp)
