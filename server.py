import json
from time import perf_counter

import requests
import uvicorn
from fastapi import FastAPI, WebSocket
import os

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI, Embedding

load_dotenv(find_dotenv())
minimax_group_id = os.getenv("MINIMAX_GROUP_ID")
minimax_api_key = os.getenv("MINIMAX_API_KEY")

embeddings_model = OpenAIEmbeddings()

client = OpenAI()
load_dotenv(find_dotenv())
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def print_json(data):
    """
    打印参数。如果参数是有结构的（如字典或列表），则以格式化的 JSON 形式打印；
    否则，直接打印该值。
    """
    if hasattr(data, 'model_dump_json'):
        data = json.loads(data.model_dump_json())

    if (isinstance(data, (list))):
        for item in data:
            print_json(item)
    elif (isinstance(data, (dict))):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)


def get_areas_information(user_question):
    content = ''
    db = FAISS.load_local(folder_path="../faiss_db", embeddings=embeddings_model, index_name="faiss_index",
                          allow_dangerous_deserialization=True)
    response = db.similarity_search(user_question)
    #retriever = db.as_retriever(search_kwargs={"k": 2})

    for i in response:
        content += i.page_content
    return content


def get_weather_information(location):
    # https://restapi.amap.com/v3/weather/weatherInfo?city=110101&key=<用户key>
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={location}&key=4c526ed13173863e7971ee2083b6a6b2"
    response = requests.get(url=url).json()
    return response


def get_completion(messages, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        seed=1024,  # 随机种子保持不变，temperature 和 prompt 不变的情况下，输出就会不变
        tool_choice="auto",  # 默认值，由 GPT 自主决定返回 function call 还是返回文字回复。也可以强制要求必须调用指定的函数，详见官方文档
        tools=[{
            "type": "function",
            "function": {
                "name": "get_areas_information",
                "description": "根据用户问题，查询相关景点信息",  #根据POI名称，获得POI的经纬度坐标
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_question": {
                            "type": "string",
                            "description": "用户提出的有关景点的问题，必须是中文",
                        },
                    },
                    #"required": ["location", "city"],
                }
            }
        },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_information",
                    "description": "查询给定位置的天气信息",  #搜索给定坐标附近的poi
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "目标位置名称",
                            },
                        },
                        #"required": ["longitude", "latitude", "keyword"],
                    }
                }
            }],
    )
    return response.choices[0].message


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_quesation = await websocket.receive_text()  #接收数据
            print(f'前端接收到的数据：{user_quesation} 开始调用大模型进行回复')


            llm_start_time = perf_counter()
            result = None
            #没有识别出内容
            if user_quesation is None:
                chat_messages = "很抱歉，没有听请你说的是什么，可以再说一次吗"
            else:
                # 调用大模型生成回复
                messages = [
                    {"role": "system", "content": "你是一个AI导游助手，可以查询景点和天气信息回答用户问题"},
                    {"role": "user", "content": user_quesation}
                ]
                response = get_completion(messages)
                messages.append(response)  # 把大模型的回复加入到对话中

                while response.tool_calls is not None:
                    # 支持一次返回多个函数调用请求，所以要考虑到这种情况
                    for tool_call in response.tool_calls:
                        args = json.loads(tool_call.function.arguments)

                        # 函数路由
                        if tool_call.function.name == "get_areas_information":
                            result = get_areas_information(**args)
                        elif tool_call.function.name == "get_weather_information":
                            result = get_weather_information(**args)

                        messages.append({
                            "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": str(result)  # 数值result 必须转成字符串
                        })
                    response = get_completion(messages)
                    messages.append(response)  # 把大模型的回复加入到对话中

                chat_messages = response.content

            llm_end_time = perf_counter()
            print(f"大模型回复：{chat_messages}")
            print(f"大模型耗时：{llm_end_time - llm_start_time}秒")

            # 将文本转换为语音
            print("开始将文本转换为语音")
            tts_start_time = perf_counter()
            tts_url = "https://api.minimax.chat/v1/tts/stream?GroupId=" + minimax_group_id
            tts_headers = build_tts_stream_headers()
            tts_body = build_tts_stream_body(chat_messages)
            response = requests.request("POST", tts_url, stream=True, headers=tts_headers, data=tts_body)
            for chunk in (response.raw):
                if chunk:
                    if chunk[:5] == b'data:':
                        data = json.loads(chunk[5:])
                        if "data" in data and "extra_info" in data:
                            if "audio" in data["data"]:
                                audio_hex = data["data"]['audio']
                                audio_bytes = bytes.fromhex(audio_hex)
                                await websocket.send_bytes(audio_bytes)
            tts_end_time = perf_counter()
            print(f"语音转换时间：{tts_end_time - tts_start_time}秒")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8765)
