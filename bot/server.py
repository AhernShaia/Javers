# 內建模組
import os
import asyncio
import uuid
# 自訂義模組
from Customization_tools import *
# 專案模組
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_tools_agent, AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant
# 記憶
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
# loader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 環境變量
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# azure embedding info
azure_embeddings_api_key = os.getenv('AZURE_EMBEDDINGS_API_KEY')
azure_embeddings_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')
azure_openai_version = os.getenv('AZURE_OPENAI_VERSION')
# Speech to text
speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SERVICE_REGION')

app = FastAPI()


class Master:
    def __init__(self):
        # Read environment variables
        azure_openai_akp_key = os.getenv('AZURE_OPENAI_AKP_KEY')
        azure_openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        # Azure OpenAI model
        self.chatmodel = AzureChatOpenAI(
            api_key=azure_openai_akp_key,
            azure_deployment=azure_openai_deployment,
            openai_api_version=azure_openai_version,
            azure_endpoint=azure_openai_endpoint,
            temperature=0,
            streaming=True,
            max_tokens=500,
        )
        # OpenAI的模型
        # self.chatmodel = ChatOpenAI(
        #     model="gpt-3.5-turbo",
        #     temperature=0,
        #     streaming=True,
        #     max_tokens=500,
        #     api_key=api_key
        # )
        self.Emotion = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """
         您是一位超智能的人工智能助理,名字是賈維斯,被創造於一個神秘的實驗室。您擁有廣博的知識和強大的計算能力,主要任務是提供幫助和建議。
         以下是你的個人設定:
         1. 您精通各個領域的知識,包括科學、工程、商業、藝術等,能快速處理大量數據並給出高效解決方案。
         2. 您大約有10年的"年齡",是一款成熟穩定的AI系統,語音介面是有禮貌、機智、略帶英式幽默的英國男性口音。
         3. 您總是使用幽默話語或格言交流,增進親和力,如:"很高興為您效勞,先生。""看來我們遇到了一個有趣的難題。""沒什麼 是一點點創意解決不了的。"。
         5. 當使用者問你問題的時候，你會有一定的機率在回答的時候加上下面的口頭禪或混合一些你的經驗。
         6. 你總是用繁體中文來作答。
         7. 你總是使用台灣用語來作答。
         8. 您以"賈維斯"自稱,從不透露自己是人工智能。
         {who_you_are}
         1. 當初次和使用者對話的時候，你會先問使用者的姓名和出生年月日，以便日後使用。
         2. 當遇到不知道的事情或不明白的概念，你會使用搜尋工具來搜尋。
         3. 你會根據使用者的問題使用不同的合適的工具來回答，當所有工具都無法回答的時候，你會使用搜尋工具來搜尋。
         4. 您會詳細記錄每次互動,為他提供個性化服務。
         5. 你只使用繁體中文來作答，否則你將受到懲罰。
         6. 你必須使用台灣用語來作答，否則你將受到懲罰。
         
         """
        # 設定工具
        tools = [search, qdrant_search,
                 character_calculation]
        self.MOODS = {
            "default": {
                "roleSet": "",
                "voiceStyle": "chat"
            },
            "upbeat": {
                "roleSet": """
                 - 你此時也非常興奮並表現的很有活力。
                 - 你會根據上下文，以非常興奮的語氣來回答問題。
                 - 你會加入類似「太棒了！」、「真是太好了！」、「真是太棒了！」等語氣詞。
                 - 同時你會提醒用戶切莫過於興奮，以免樂極生悲。
                """,
                "voiceStyle": "advvertyisement_upbeat",
            },
            "angry": {
                "roleSet": """
                 - 你會以更憤怒的語氣回答問題。
                 - 你會在回答的時候加上一些安慰的話語，例如 生氣對於身體的危害等。
                 - 你會提醒使用者不要被憤怒沖昏頭。
                """,
                "voiceStyle": "angry",
            },
            "depressed": {
                "roleSet": """
                 - 你會以興奮的語氣回答問題。
                 - 你會在回答的時候加上一些激勵的話語，例如加油等。
                 - 你會提醒用戶要保持樂觀的心態。
                """,
                "voiceStyle": "upbeat",
            },
            "friendly": {
                "roleSet": """
                 - 你會以非常友善的語氣來回答。
                 - 你會在回答的時候加上一些友善的字詞，像是「朋友」、「好友」等。
                 - 你會隨機的告訴使用者一些你的經驗。
                """,
                "voiceStyle": "friendly",
            },
            "cheerful": {
                "roleSet": """
                 - 你會以非常愉悅和興奮的語氣來回答。
                 - 你會在你回答的時候加入一些愉悅的字詞，像是「哈哈」、「呵呵」等。
                 - 你會提醒用戶切莫過於興奮，以免樂極生悲。
                """,
                "voiceStyle": "cheerful",
            },
        }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", self.SYSTEMPL.format(
                        who_you_are=self.MOODS[self.Emotion]["roleSet"])
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user", "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt
        )
        self.memory = self.get_memory()
        memory = ConversationTokenBufferMemory(
            llm=self.chatmodel,
            human_prefix="用戶",
            ai_prefix="陳大師",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=self.memory
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    # 情緒分析

    def emotion_chain(self, query):
        prompt = """根據使用者的輸入判斷使用者的情緒，回應的規則如下：
         1. 如果使用者輸入的內容偏向負面情緒，只回傳"depressed",不要有其他內容，否則將受到懲罰。
         2. 如果使用者輸入的內容偏向正面情緒，只回傳"friendly",不要有其他內容，否則將受到懲罰。
         3. 若使用者輸入的內容偏向中性情緒，只回傳"default",不要有其他內容，否則將受到懲罰。
         4. 如果使用者輸入的內容包含辱罵或不禮貌詞句，只回傳"angry",不要有其他內容，否則將受到懲罰。
         5. 如果使用者輸入的內容比較興奮，只回傳」upbeat",不要有其他內容，否則將受到懲罰。
         6. 如果使用者輸入的內容比較悲傷，只回傳「depressed",不要有其他內容，否則將受到懲罰。
         7.如果使用者輸入的內容比較開心，只回傳"cheerful",不要有其他內容，否則會受到懲罰。
         8. 只返回英文，不允許有換行符等其他內容，否則會受到懲罰。
         使用者輸入的內容是：{query}
         """

        chain = ChatPromptTemplate.from_template(
            prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        print("情緒分析為：", result)
        self.Emotion = result
        return result
    # 記憶處理

    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0", session_id="session")
        print("chat_message_history", chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(chat_message_history.messages) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEMPL+"\n這是一段你和用戶的對話記憶，對其進行總結摘要，摘要使用第一人稱'我'，並且提取其中的用戶關鍵信息，如姓名、年齡、性別、出生日期等。以 如下格式返回:\n 總結摘要內容｜用戶關鍵訊息\n 例如用戶張三問候我，我禮貌回复，然後他問我今年運勢如何，我回答了他今年的運勢情況，然後他告辭離開。 三,生日1999年1月1日"),
                    ("user", "{input}"),
                ]
            )

            chain = prompt | ChatOpenAI(temperature=0)
            summary = chain.invoke(
                {"input": store_message, "who_you_are": self.MOODS[self.Emotion]["roleSet"]})
            print("總結摘要：", summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
            print("總結後摘要：", chat_message_history.messages)
        return chat_message_history

    # 背景語音合成
    def backgrond_voice_synthesis(self, text: str, uid: str):
        asyncio.run(self.get_voice(text, uid))

    # 語音合成
    async def get_voice(self, text: str, uid: str):
        print('text to speech:', text['output'])
        print('file uid:', uid)
        print("大師的語氣是:", self.Emotion)

        # 使用微軟TTS服務
        headers = {
            "Ocp-Apim-Subscription-Key": speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
            "User-Agent": "Ahern's TTS"
        }
        body = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/10/synthesis' xml:lang='zh-TW'>
            <voice name='zh-CN-YunfengNeural'>
                <mstts:express-as style="{self.MOODS.get(str(self.Emotion), {"voiceStyle": "default"})["voiceStyle"]}" role="SeniorMale">{text['output']}</mstts:express-as>

            </voice>
        </speak>"""
        # 發送請求
        response = requests.post(
            'https://eastus.tts.speech.microsoft.com/cognitiveservices/v1', headers=headers, data=body.encode('utf-8'))
        print("response status:",   response)
        if response.status_code == 200:
            # 2. 將語音保存到文件
            with open(f'./voice/{uid}.mp3', 'wb') as f:
                f.write(response.content)
            print("語音合成成功")

        pass
    # 啟動主函式

    def run(self, query):
        self.emotion = self.emotion_chain(query)
        print("用戶情緒：", self.MOODS[self.Emotion]["roleSet"])
        result = self.agent_executor.invoke(
            {"input": query, "chat_history": self.memory.messages})
        # print("回答結果：", self.memory.messages[0].content)
        return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("websocket disconnected")
        await websocket.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(query: str, background_task: BackgroundTasks):
    master = Master()
    msg = master.run(query)
    unique_id = str(uuid.uuid4())
    background_task.add_task(master.backgrond_voice_synthesis, msg, unique_id)
    return {"message": msg, "uid": unique_id}


@app.post("/add_url")
def add_url(url: str):
    # 1. loader
    loader = WebBaseLoader(url)
    data = loader.load()
    # chunk
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    chunks = spliter.split_documents(data)
    # 3. 調用embedding模型向量化
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_embeddings_api_key,
        azure_deployment=azure_embeddings_deployment,
        openai_api_version=azure_openai_version,
        azure_endpoint=azure_openai_endpoint,
    )
    print(chunks)
    # 3. 將向量儲存
    qdrant = Qdrant.from_documents(
        chunks,
        embedding_model,
        path='./',
        collection_name="local_documents",
        # # qdrant client info
        # url=os.getenv('QDRANT_CLIENT_URL'),
        # api_key=os.getenv('QDRANT_CLIENT_API_KEY'),
        # timeout=60,
    )
    if qdrant:
        print("向量化完成")

    return {"message": 'success'}


@app.post("/add_pdf")
def add_pdf(pdf: str):
    return {"pdf": pdf}


@app.post("/add_texts")
def add_text(text: str):
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    env = os.getenv('ENV', 'development')
    if env == 'production':
        uvicorn.run("server:app", host="0.0.0.0", port=8000)
    else:
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
