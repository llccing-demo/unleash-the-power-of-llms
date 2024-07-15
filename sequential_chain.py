import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain 

load_dotenv()
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

# defining the original input
biography = "He is an American author of thriller fiction, best known for his Robert Langdon series. \
          He has sold over 200 million copies of his books, which have been translated into 56 \
          languages. His other works include Angels & Demons, The Lost Symbol, Inferno, and Origin. \
          He is a New York Times best-selling author and has been awarded numerous awards for his \
          writing."

prompt_1 = ChatPromptTemplate.from_messages(
    [("user", "Summarize this biography in one sentence: \n\n{biography}")],
)
chain_1 = LLMChain(llm=model, prompt=prompt_1, output_key="one_line_biography")
#chain_1 = prompt_1 | model

prompt_2 = ChatPromptTemplate.from_messages(
    [("user", "Can you tell teh author's name in this biography: \n\n{one_line_biography}")]
)
chain_2 = LLMChain(llm=model, prompt=prompt_2, output_key="author_name")
#chain_2 = prompt_2 | model

prompt_3 = ChatPromptTemplate.from_messages(
    [("user", "Tell the name of the highest selling book of this author: \n\n{author_name}")]
)
chain_3 = LLMChain(llm=model, prompt=prompt_3, output_key="book")
#chain_3 = prompt_3 | model

prompt_4 = ChatPromptTemplate.from_messages(
    [("user", "Write a follow-up response to the following summary of the highest-selling book of the author: \n\nAuthor: {author_name}\n\nBook: {book}")]
)
chain_4 = LLMChain(llm=model, prompt=prompt_4, output_key="summary")
#chain_4 = prompt_4 | model

sequential_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4],
    input_variables=["biography"],
    output_variables=["one_line_biography", "author_name", "summary"],
    verbose=True,
)

resp = sequential_chain.invoke(biography)
print(resp)
