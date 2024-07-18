import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

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

while True:
  question = input("enter your query: ")
  if question == 'exit':
    break
  result = question_answering({"question": 'Answer only in the context of the document provided.' + question})
  print(result['answer'])


# sample of answer 
"""
>> from langchain_community.document_loaders import PyPDFLoader
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/v0.2/docs/versions/v0_2/>
  warn_deprecated(
enter your query: what is seo
/home/ubuntu/projects/educative/unleash-the-power-of-llms/.venv/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:139: LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 0.3.0. Use invoke instead.
  warn_deprecated(
SEO refers to the techniques and strategies used to optimize individual web pages in order to improve their search engine rankings and drive more relevant traffic to the site. It includes both On-Page SEO, which focuses on tactics applied to a given website page, and Technical SEO, which ensures the full potential of a website is unlocked.
enter your query: how to do seo
To perform SEO, you need to focus on three fundamental areas: on-page SEO, off-page SEO, and technical SEO.

1. **On-Page SEO**: This involves optimizing individual pages on your website to rank higher and earn more relevant traffic. Key aspects include:
   - Using relevant keywords in your content.
   - Crafting compelling and keyword-rich titles and meta descriptions.
   - Ensuring content quality and relevance.
   - Optimizing images with alt tags.
   - Making sure the website is mobile-friendly.

2. **Off-Page SEO**: This focuses on actions taken outside of your website to impact your rankings within search engine results pages (SERPs). Key aspects include:
   - Building high-quality backlinks from reputable sites.
   - Social media marketing to drive traffic and engagement.
   - Influencer marketing and guest blogging.

3. **Technical SEO**: This involves improving the technical aspects of your website to ensure it's easy for search engines to crawl and index. Key aspects include:
   - Ensuring fast page load speeds.
   - Creating an XML sitemap.
   - Using HTTPS to secure the site.
   - Fixing any crawl errors and broken links.

By focusing on these three key areas, you can enhance your website's visibility and performance in search engine results.
  """