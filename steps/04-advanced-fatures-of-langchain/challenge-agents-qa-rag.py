import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.agents import initialize_agent, Tool, AgentType


load_dotenv()
#########   LLM Chain     ###############
# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERPAPI_API_KEY")
os.environ['SERPAPI_API_KEY']=serp_api_key
url = "https://oneapi.gptnb.me/v1"
model = ChatOpenAI(
    #model="claude-3-sonnet-20240229",
    model="gpt-4o",
    #model="gemini-1.5-pro-latest",
    openai_api_key=api_key,
    openai_api_base=url,
)

loader = PyPDFLoader('./3-Fundamentals-Of-SEO.pdf')
mypdf = loader.load()

document_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 300,
  chunk_overlap = 70
)

docs = document_splitter.split_documents(mypdf)

embeddings = OpenAIEmbeddings(
  model="text-embedding-3-large",
  #model="gemini-1.5-pro-latest",
  openai_api_key=api_key,
  openai_api_base=url,
)

persist_directory = 'db'

my_database = Chroma.from_documents(
  documents=docs,
  embedding=embeddings,
  persist_directory=persist_directory
)

retaining_memory = ConversationBufferWindowMemory(
  memory_key='chat_history',
  k=5,
  return_messages=True
)

question_answering = ConversationalRetrievalChain.from_llm(
  llm = model,
  retriever=my_database.as_retriever(),
  memory=retaining_memory
)

# tools
tools = [
  Tool(
    name='Knowledge Base',
    func=question_answering.run,
    description=(
      'use this tool when answering questions related to the seo.'
    )
  )
]

# initializing the agent
agent = initialize_agent(
  agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
  tools=tools,
  llm=model,
  verbose=True,
  max_iterations=3,
  early_stopping_method='generate',
  memory=retaining_memory
)

while True:
  question = input("enter your query: ")
  if question == 'exit':
    break
  result = agent.run(question)
  print(result['answer'])
