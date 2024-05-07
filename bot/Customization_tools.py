import os
import requests
import json
# langchain
from langchain.agents import tool
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
# serpapi info
search_api_key = os.getenv('SERPAPI_API_KEY')

# azure embedding info
azure_embeddings_api_key = os.getenv('AZURE_EMBEDDINGS_API_KEY')
azure_embeddings_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_version = os.getenv('AZURE_OPENAI_VERSION')

character_calculation_api_key = os.getenv('CHARACTER')


@tool
def test():
    """This is a test tool."""
    return "test tool"


@tool
def search(query: str):
    """Searches the web for the given query.
        只有在使用者問題無法回答或需要了解實時資訊時才會使用這個工具。
    """
    # 搜索引擎設定為 Google,
    # 地理位置設定為台灣,這將影響搜索結果的地理相關性
    # 搜索結果的語言設定為繁體中文
    params = {
        "engine": "google",
        "gl": "tw",
        "hl": "tw",
    }

    serp = SerpAPIWrapper(params=params, serpapi_api_key=search_api_key)
    result = serp.run(query)
    print("search result:", result)
    return result


@tool
def qdrant_search(query: str):
    """當用戶詢問法律相關問題時,使用這個工具來搜索相關的法律文件。"""
    qdrant_client = QdrantClient(
        # qdrant client info
        url=os.getenv('QDRANT_CLIENT_URL'),
        api_key=os.getenv('QDRANT_CLIENT_API_KEY'),
        https=True,
    )
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_embeddings_api_key,
        azure_deployment=azure_embeddings_deployment,
        openai_api_version=azure_openai_version,
        azure_endpoint=azure_openai_endpoint,
    )
    client = Qdrant(
        client=qdrant_client,
        embeddings=embedding_model,
        collection_name="legalassistant",
    )
    retriever = client.as_retriever(search_kwargs={"k": 3})
    result = retriever.invoke(query)
    return result

# 命理算命


@tool
def character_calculation(query: str):
    """當用戶詢問命理算命相關問題時，使用這個工具，需要用戶輸入: 姓名、出生年份、出生月份、出生日期、性別，如果缺少 姓名、出生年份、出生月份、出生日期、性別，則不可以使用，否則將受到懲罰。"""
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """你是參數查詢助手，根據使用者輸入內容找出相關的參數並按json格式返回。JSON欄位如下： -"api_ke":"{character_calculation_api_key}", - "name":"姓名", - "sex ":"性別，0表示男，1表示女，依姓名判斷", - "type":"日曆類型，0農曆，1公里，預設1"，- "year":"出生年份例：1998", - "month":"出生月份例8", - "day":"出生日期，例：8", - "hours":"出生小時例14", - "minute":"0"，如果沒有找到 相關參數，則需要提醒使用者告訴你這些內容，只回傳資料結構，不要有其他的評論，使用者輸入:{query}""")

    parser = JsonOutputParser()
    prompt = prompt.partial(from_instruction=parser.get_format_instructions())
    print("八字測算Prompt：", prompt)
    chain = prompt | ChatOpenAI(temperature=0) | parser
    data = chain.invoke({"query": query})
    print("八字測算data：", data)
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print("----------------------返回結果----------------------")
        print(result.json())
        print("----------------------返回結果----------------------")
        try:
            json = result.json()
            returnstring = f"八字為:{json["data"]["bazi_info"]["bazi"]}"
            return returnstring
        except Exception as e:
            return "無法獲取八字資訊，請檢察用戶的  姓名、出生年份、出生月份、出生日期、性別是否正確。"
    else:
        return "無預期錯誤，請稍後再試。"

# API 暫時無法使用
# @tool
# def divination(query: str):
#     """只有使用者想要占卜抽籤的時候才會使用這個工具。"""
#     url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
#     result = requests.post(
#         url, data={"api_key": character_calculation_api_key})
#     if result.status_code == 200:
#         print("====返回数据=====")
#         print(result.json())
#         returnstring = json.loads(result.text)
#         image = returnstring["data"]["image"]
#         print("卦图片:", image)
#         return returnstring
#     else:
#         return "無預期錯誤，請稍後再試。"


# API 暫時無法使用
# @tool
# def dream_interpretation(query: str):
#     """只有使用者想要解夢的時候才會使用這個工具。"""
#     url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
#     data = {
#         "api_key": character_calculation_api_key,
#         "title_zhougong": query  # 將 query 變數作為 title_zhougong 的值
#     }
#     result = requests.post(url, data=data)
#     if result.status_code == 200:
#         print("----------------------返回結果----------------------")
#         print(result.json())
#         print("----------------------返回結果----------------------")
#         try:
#             json = result.json()
#             returnstring = f"解夢結果:{json["data"]["message"]}"
#             return returnstring
#         except Exception as e:
#             return "無法獲取解夢資訊，請檢察用戶輸入的內容是否正確。"
#     else:
#         return "無預期錯誤，請稍後再試。"

@tool
def jiemeng(query: str):
    """只有使用者想要解夢的時候才會使用這個工具,需要輸入使用者夢境的內容，如果缺少使用者夢境的內容則不可用。"""
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    LLM = OpenAI(temperature=0)
    prompt = PromptTemplate.from_template("根據內容提取1個關鍵字，只回傳關鍵字，內容為:{topic}")
    prompt_value = prompt.invoke({"topic": query})
    keyword = LLM.invoke(prompt_value)
    print("擷取的關鍵字:", keyword)
    result = requests.post(
        url, data={"api_key": character_calculation_api_key, "title_zhougong": keyword})
    if result.status_code == 200:
        print("----------------------返回結果----------------------")
        print(result.json())
        print("----------------------返回結果----------------------")
        returnstring = json.loads(result.text)
        return returnstring
    else:
        return "技术错误，请告诉用户稍后再试。"
