"""
将文本转换成语音的函数
"""
import asyncio
import json
import os

from dotenv import load_dotenv, find_dotenv
import requests

load_dotenv(find_dotenv())
minimax_group_id = os.getenv("MINIMAX_GROUP_ID")
minimax_api_key = os.getenv("MINIMAX_API_KEY")


def build_tts_stream_headers() -> dict:
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + minimax_api_key,
    }
    return headers


def build_tts_stream_body(text: str) -> dict:
    body = json.dumps({
        "text": text,
        "voice_id": "female-shaonv",
        "model": "speech-01",
        "format": "mp3",
    })
    return body


async def text_to_speech(text: str) -> bytes:
    tts_url = "https://api.minimax.chat/v1/tts/stream?GroupId=" + minimax_group_id
    tts_headers = build_tts_stream_headers()
    tts_body = build_tts_stream_body(text)
    response = requests.request("POST", tts_url, stream=True, headers=tts_headers, data=tts_body)
    for chunk in response.raw:
        if chunk:
            if chunk[:5] == b'data:':
                data = json.loads(chunk[5:])
                if "data" in data and "extra_info" in data:
                    if "audio" in data["data"]:
                        audio_hex = data["data"]['audio']
                        audio_bytes = bytes.fromhex(audio_hex)
                        return audio_bytes

async def text(dfsd):
    audio_bytes = await text_to_speech(dfsd)
    print(audio_bytes)

if __name__ == '__main__':
    dfsd = "你好！我可以帮助你查询景点信息和天气情况。如果你有任何相关的问题，比如想了解某个景点的详细信息或者某个地方的天气预报，随时可以问我！"
    response = asyncio.run(text(dfsd))
    print(response)