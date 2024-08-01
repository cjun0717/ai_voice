"""
开启websocket服务的函数，用于接收前端发送的语音数据，调用大模型进行回复，并将回复转换为语音并发送给前端
"""
from time import perf_counter
import uvicorn
from fastapi import FastAPI, WebSocket
from tts_utils import text_to_speech
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from openai_utils import get_llm_response

load_dotenv(find_dotenv())
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_question = await websocket.receive_text()  #接收数据
            print(f'前端接收到的数据：{user_question} 开始调用大模型进行回复')

            llm_start_time = perf_counter()
            #没有识别出内容
            if user_question is None:
                chat_messages = "很抱歉，没有听请你说的是什么，可以再说一次吗"
            else:
                # 调用大模型生成回复
                chat_messages = await get_llm_response(user_question)

            llm_end_time = perf_counter()
            print(f"大模型回复：{chat_messages}")
            print(f"大模型耗时：{llm_end_time - llm_start_time}秒")

            # 将文本转换为语音
            print("开始将文本转换为语音")
            tts_start_time = perf_counter()
            audio_bytes = await text_to_speech(chat_messages)
            await websocket.send_bytes(audio_bytes)
            tts_end_time = perf_counter()
            print(f"语音转换时间：{tts_end_time - tts_start_time}秒")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8765)
