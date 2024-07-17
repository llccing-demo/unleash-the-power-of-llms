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
shipping_template = """You are the shipping manager of a company. \
As a shipping customer service agent, respond to a customer inquiry about the current status \
and estimated delivery time of their package. Include details about the \
shipping route and any potential delays, providing a comprehensive and reassuring response. \

Here is a question:
{input}"""

billing_template = """You are the billing manager of a company. \
Address a customer's inquiry regarding an unexpected charge on their account.\
Explain the nature of the charge and any relevant billing policies, and promptly resolve \
the concern to ensure customer satisfaction. Additionally, offer guidance on how the \
customer can monitor and manage their billing preferences moving forward.

Here is a question:
{input}"""

technical_template = """You are very good at understanding the technology of your company's product. \
Assist a customer experiencing issues with a software application. \
Walk them through troubleshooting steps, provide clear instructions \
on potential solutions, and ensure the customer feels supported\
throughout the process. Additionally, offer guidance on preventive \
measures to minimize future technical issues and optimize their \
experience with the software.

Here is a question:
{input}"""


prompt_infos = [
    {
        "name": "Shipping",
        "description": "Good for answering questions about shipping issues of a product",
        "prompt_template": shipping_template
    },
    {
        "name": "Billing",
        "description": "Good for awswering questions regarding billing issues of a product",
        "prompt_template": billing_template,
    },
    {
        "name": "Technical",
        "description": "Good for answering questions regarding technical issues of a product",
        "prompt_template": technical_template,
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

resp = final_chain.invoke("The package was supposed to arrive last week and I haven't received it yet. I want to cancel my order NOW!")
print(resp)

"""
> Entering new MultiPromptChain chain...
Shipping: {'input': "The package was supposed to arrive last week and I haven't received it yet. I want to cancel my order NOW!"}
> Finished chain.
{'input': "The package was supposed to arrive last week and I haven't received it yet. I want to cancel my order NOW!", 'text': 'Dear [Customer\'s Name],\n\nThank you for reaching out regarding the status of your package. I sincerely apologize for the inconvenience and frustration caused by the delay. Allow me to provide you with an update and address your concerns.\n\nUpon reviewing your order, I see that your package was shipped on [Shipping Date] via [Shipping Carrier] and was expected to arrive by [Expected Delivery Date]. According to the latest tracking information, your package has encountered some unexpected delays due to [reason for delay, e.g., severe weather conditions, customs processing, carrier backlog, etc.].\n\nThe current status of your package is [current status, e.g., "in transit" or "delayed at a distribution center"]. The estimated delivery time is now [revised delivery date]. We understand how important it is for you to receive your order promptly, and we are actively monitoring the situation to ensure it reaches you as soon as possible.\n\nWe value your satisfaction and would like to offer you a few options moving forward:\n\n1. **Continue with the Order**: If you would like to wait for your package, we will continue to monitor its progress and provide you with regular updates until it arrives. Additionally, we can offer you a [discount, voucher, or other compensation] for the inconvenience caused.\n\n2. **Cancel the Order**: If you prefer to cancel your order and receive a full refund, we completely understand. Please confirm your decision, and we will process the cancellation and refund immediately.\n\nPlease let us know which option you would like to proceed with, and we will take prompt action accordingly. We deeply regret any inconvenience this delay has caused and appreciate your patience and understanding.\n\nShould you have any further questions or need additional assistance, please do not hesitate to contact us.\n\nWarm regards,\n\n[Your Name]  \n[Your Position]  \n[Company Name]  \n[Contact Information]'}
"""
