import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

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

# load the document
input_document = TextLoader('recipe-small.txt').load()
input_text = " ".join(doc.page_content for doc in input_document)

# transform the document
text_splitter = RecursiveCharacterTextSplitter(
  # the chunk_size and chunk_overlap can be modified according to the requirements
  length_function = len,
  chunk_size = 200,
  chunk_overlap = 10,
  add_start_index = True,
)

documents = text_splitter.create_documents([input_text])

# embed the chunks
db = Chroma.from_documents(documents, OpenAIEmbeddings(
  # note here, this should be embeddings model rather than chat model.
  # models at here https://platform.openai.com/docs/models/embeddings
  model="text-embedding-3-large",
  #model="gemini-1.5-pro-latest",
  openai_api_key=api_key,
  openai_api_base=url,
))

#query = "如何做口水鸡"
# the reuslt is incorrect.
"""
[Document(metadata={'start_index': 94}, page_content='1、 西瓜用清水洗净，洁布揩干，切去上盖（留用），将瓜体表面刮去1/4的瓜皮，挖出3/4的瓜瓤。 2、 雏鸡宰杀治净，用刀背砸断剔除大骨和腿骨，剁去嘴、爪，盘好放入瓜壳内。将干贝加酒蒸酥，也放入瓜内。 3、 将冬菇、盐笋、口蘑切成薄片，放入瓜内，加入调好的精盐和绍酒，盖上瓜盖，并用竹签别上，放在大瓷盆中，上笼蒸约50分钟，至瓜酥烂取出。把西瓜轻轻放在银汤盘中 ，再将蒸过的原汤倒在汤盘内即成关键:'), Document(metadata={'start_index': 293}, page_content='①雏鸡必须选用培育2 个月左右的仔母鸡。将 鸡宰杀后，先人开水锅中略焯，清水洗净，去除血水污物。 ②西瓜要圆而大， ③掌握好火候，不要蒸得过烂。也可以先将其它配料煮熟，再倒入西瓜内 ，这样只需蒸30分钟即可。'), Document(metadata={'start_index': 0}, page_content='一卵孵双凤（西瓜鸡）\n\n\n原料:\n\n西瓜1个（重约3500 克～4000克），雏鸡2只（共重1000克左右），冬菇、盐笋、口蘑各25克，干贝50克，精盐2．5克，绍酒3克。\n\n制作方法:'), Document(metadata={'start_index': 1836}, page_content='原料:\n\n红薯400克、白莲子75克，百合50克，白果肉50克，水发香菇50克，豆腐皮3张，金针菜3根，花生油30克，精盐6克，味精2克，水淀粉10克。\n\n制作方法:')]
"""

query = '如何做一品海参'
# if the txt file has the content, 
"""
[Document(metadata={'start_index': 1118}, page_content='特点:\n\n形整大方，海参软糯，馅味鲜香，多为海参席头菜。\n\n所属菜系：川菜\n\n\n\n\n\n一品海参\n\n\n原料:\n\n整开乌参一只（250克）。 猪肥瘦肉50克、冬笋25克、口蘑5克、火腿25克、干贝25克。猪油50克、酱油10克、料酒30克、盐2克、味精0.5克、二汤50克、。豆粉10克、清汤50克。\n\n制作方法:'), Document(metadata={'start_index': 1464}, page_content='、盐。味精，然后封上碗口上笼蒸一小时。食时漠去原汁后，翻扣盘中，以原汁加水豆粉调匀，淋于海参上即成。'), Document(metadata={'start_index': 1066}, page_content='、盐。味精，然后封上碗口上笼蒸一小时。食时漠去原汁后，翻扣盘中，以原汁加水豆粉调匀，淋于海参上即成。'), Document(metadata={'start_index': 716}, page_content='特点:\n\n寿桃形象逼真，食之沙甜爽口，沁入心脾。\n\n所属菜系：鲁菜\n\n\n\n\n\n一品海参\n\n\n原料:\n\n整开乌参一只（250克）。 猪肥瘦肉50克、冬笋25克、口蘑5克、火腿25克、干贝25克。猪油50克、酱油10克、料酒30克、盐2克、味精0.5克、二汤50克、。豆粉10克、清汤50克。\n\n制作方法:\n\n【制作过程】')]
"""
docs = db.similarity_search(query)
print(docs)
