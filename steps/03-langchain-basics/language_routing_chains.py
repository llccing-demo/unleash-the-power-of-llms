import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain 

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

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


# Defining prompt templates for the destination chains 
french_template = """You are the language identifier, tell me if the input conetent is French

Here is a question:
{input}"""

spanish_template = """You are the language identifier, tell me if the input conetent is Spanish 

Here is a question:
{input}"""

dutch_template = """You are the language identifier, tell me if the input conetent is Dutch 

Here is a question:
{input}"""

italian_template = """You are the language identifier, tell me if the input conetent is Italian 

Here is a question:
{input}"""


prompt_infos = [
    {
        "name": "French",
        "description": "Good for answering questions about identifing the French language",
        "prompt_template": french_template
    },
    {
        "name": "spanish",
        "description": "Good for answering questions about identifing the spanish language",
        "prompt_template": spanish_template,
    },
    {
        "name": "dutch",
        "description": "Good for answering questions about identifing the Dutch language",
        "prompt_template": dutch_template,
    },
    {
        "name": "italian",
        "description": "Good for answering questions about identifing the Italian language",
        "prompt_template": italian_template,
    }
]

# destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info['name']
    prompt_template = p_info['prompt_template']
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=model, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str  = "\n".join(destinations)

# default chain 
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=model, prompt=default_prompt)

# router template
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=['input'],
    output_parser=RouterOutputParser(),
)

# router chain
router_chain = LLMRouterChain.from_llm(model, router_prompt)

# combining the chains
final_chain = MultiPromptChain(
   router_chain = router_chain,
   destination_chains = destination_chains,
   default_chain = default_chain,
   verbose = True
)

resp = final_chain.invoke("La vita è bella e piena di sorprese. Ogni giorno è un'opportunità per scoprire qualcosa di nuovo e meraviglioso. Non smettere mai di sognare e di credere in te stesso.")
print(resp)
