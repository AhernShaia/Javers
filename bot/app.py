from linebot.models.events import MemberJoinedEvent
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
import uuid
from linebot import LineBotApi
from linebot.models import *
import linebot.v3.messaging
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    MessagingApi,
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3 import (
    WebhookHandler
)
from fastapi import FastAPI, Request, HTTPException
import os
# 環境變量
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
configuration = Configuration(access_token=access_token)
handler = WebhookHandler(channel_secret)


# 監聽所有來自 /callback 的 Post Request


@app.post("/callback")
async def callback(request: Request):
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode("utf-8")

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(
            status_code=400, detail="Invalid signature. Please check your channel access token/channel secret.")

    return 'OK'


# 處理事件
@handler.add(MessageEvent)
def handle_message(event):
    print("event type is:", event.type)
    # 語音事件
    if event.message.type == 'audio':
        UserSendAudio = LineBotApi(
            os.getenv('LINE_CHANNEL_ACCESS_TOKEN')).get_message_content(event.message.id)
        unique_id = str(uuid.uuid4())
        path = './static/' + unique_id + '.m4a'
        with open(path, 'wb') as fd:
            for chunk in UserSendAudio.iter_content():
                fd.write(chunk)
    # 文字事件
    elif event.type == 'message':
        with ApiClient(configuration) as api_client:
            # ---Line 前端加載動畫 start---
            api_instance = linebot.v3.messaging.MessagingApi(api_client)
            show_loading_animation_request = linebot.v3.messaging.ShowLoadingAnimationRequest(
                chatId=event.source.user_id, loadingSeconds=5)
            api_response = api_instance.show_loading_animation(
                show_loading_animation_request)
            print("Messaging Api->顯示載入動畫的回應：\n")
            # ---Line 前端加載動畫 end---
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=event.message.text)]
                )
            )


@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name}歡迎加入,我是您的私人助理,我叫賈維斯,有什麼問題都可以問我哦！')
    ReplyMessageRequest(
        reply_token=event.reply_token,
        messages=[TextMessage(text=message)]
    )


# 啟動 FastAPI 伺服器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
