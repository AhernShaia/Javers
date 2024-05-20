import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('WHISPER_API_KEY')
api_version = os.getenv('WHISPER_API_VERSION')
azure_endpoint = os.getenv('WHISPER_ENDPOINT')

# 只需要傳入檔案路徑即可


def speech_to_text(file_path):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )

    # This will correspond to the custom name you chose for your deployment when you deployed a model."
    deployment_id = "whisper"

    result = client.audio.transcriptions.create(
        file=open(file_path, "rb"),
        model=deployment_id
    )

    return result.text
