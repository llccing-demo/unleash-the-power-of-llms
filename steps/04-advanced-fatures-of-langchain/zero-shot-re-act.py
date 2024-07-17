
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain 
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

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

# loading tools 
tools = load_tools(
  ['serpapi', 'llm-math'],
  llm=model
)

agent = initialize_agent(
  tools,
  llm=model,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
)

res = agent.invoke("特朗普被谁刺杀了，特朗普还活着吗?")
# res = agent.invoke("今天大连的天气怎么样?")


"""
> Entering new AgentExecutor chain...
特朗普（Donald Trump）是前美国总统，关于他是否被刺杀以及他是否还活着是一个需要最新信息的问题。为了获取最新的准确信息，我需要进行搜索 。

Action: Search
Action Input: 特朗普被刺杀了吗，特朗普还活着吗
Observation: ['在美国宾夕法尼亚州举行的集会上，特朗普遭到枪手袭击，这名袭击者被特勤局成员开枪击毙。联邦调查局表示，他们正在调查这起暗杀企图，并公布了嫌疑人 ...', '美国总统拜登在特朗普遇刺后向全国发表讲话，敦促美国人“降温”并通过投票箱而不是通过子弹解决分歧。“我们不能让这种暴力正常化。”', '美国前总统唐纳德·特朗普(Donald Trump)在竞选场合遭枪击事件，星期天(7月14日)一早就成为中国微博热搜的“爆词”，多家主要媒体也大篇幅报道此事。', '美国前总统唐纳德·特朗普在宾夕法尼亚州集会上遭枪击后被护送下台，耳朵和脸颊上沾满鲜血，目前“安全”了。', '特朗普险遭枪杀让马斯克的态度发生转变。他毫不犹豫表明了支持特朗普重返白宫的立场，并调动社交媒体影响力，频繁表达情绪和事实——或是缺乏事 实依据的 ...', '唐纳德·约翰·特朗普（英语：Donald John Trump；1946年6月14日—），美国政治人物，第45任美国总统。从政前为企业家、媒体名人。', '美国联邦调查局(FBI)星期日(7月14日)凌晨表示，已确认20岁的马修·克鲁克斯(Matthew Crooks)是星期六刺杀美国前总统唐纳德·特朗普企图中 的“涉案人员”， ...', '美国总统候选人特朗普在针对他的枪击中幸免于难。事实上，在美国历史上先后有多位政治人物遭遇暗杀，其中一些总统或参 议员的遇刺改变了这个国家发展的 ...', '在当地时间13日经历“未遂刺杀”事件后，美国前总统特朗普14日接受《纽约邮报》采访时回忆了事件细节， 并对美国总统拜登事后打来电话表示“感谢”。', '根据民调，美国前总统唐纳德·特朗普（川普，Donald Trump）有望在11月的总统大选中重掌权力，但现在还难分胜负。 无论谁获胜，选举结果都将波及全世界， ...']
Thought:根据搜索结果，美国前总统唐纳德·特朗普（Donald Trump）确实在宾夕法尼亚州的一场集会上遭到枪手袭击，袭击者被特勤局成员开枪击毙。特朗普在事件中幸免于难，并在之后接受采访时表示他目前是安全的。因此，特朗普还活着。

Final Answer: 特朗普在宾夕法尼亚州的一场集会上遭到枪手袭击，但他幸免于难，目前还活着。

> Finished chain.
"""