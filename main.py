from speech_to_text import speech_to_text
from fastapi import FastAPI, WebSocket
import os
import ffmpeg
import uuid
from text_to_speech import text_to_speech
from fastapi.middleware.cors import CORSMiddleware
from get_llm_response import get_llm_response

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#从前端接受文件路径
RECEIVE_FOLDER = 'receive'
#向前端发送文件路径
SEND_FOLDER = 'send'


# 确保上传文件夹存在
if not os.path.exists(RECEIVE_FOLDER):
    os.makedirs(RECEIVE_FOLDER)
if not os.path.exists(SEND_FOLDER):
    os.makedirs(SEND_FOLDER)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            #前端接收到的音频数据格式为ogg
            audio_filename = os.path.join(RECEIVE_FOLDER, f"{uuid.uuid4()}.ogg")
            data = await websocket.receive_bytes() #接收音频数据

            with open(audio_filename, "wb") as f:
                f.write(data)

            #使用ffmpeg转换
            audio_newname = os.path.join(RECEIVE_FOLDER, f"{uuid.uuid4()}.wav")
            ffmpeg.input(audio_filename).output(audio_newname).run()

            # 语音识别
            text = speech_to_text(audio_newname)

            # 调用大模型生成回复
            # 创建消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个智能聊天机器人"
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            response_texe = get_llm_response(messages)


            # 将文本转换为语音
            text_to_speech(response_texe, "output.mp3")


            # 发送语音回复给前端
            with open("output.mp3", "rb") as audio_file:
                await websocket.send_bytes(audio_file.read())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
