"""
使用大模型的Function Calling功能回答用户问题的函数，包括工具函数的定义
"""
import asyncio
import json

import requests
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = OpenAI()
embeddings_model = OpenAIEmbeddings()

#查询景点信息的工具函数
async def get_areas_information(user_question):
    content = ''
    db = FAISS.load_local(folder_path="../faiss_db", embeddings=embeddings_model, index_name="faiss_index",
                          allow_dangerous_deserialization=True)
    response = db.similarity_search(user_question)

    for i in response:
        content += i.page_content
    return content

#查询天气信息的工具函数
async def get_weather_information(location):
    # https://restapi.amap.com/v3/weather/weatherInfo?city=110101&key=<用户key>
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={location}&key=api_key"
    response = requests.get(url=url).json()
    return response

#调用一次大模型的回复
async def get_completion(messages, model="gpt-4o-mini"):
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
                "description": "根据用户问题，查询相关景点信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_question": {
                            "type": "string",
                            "description": "用户提出的有关景点的问题，必须是中文",
                        },
                    },

                }
            }
        },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_information",
                    "description": "查询给定位置的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "目标位置名称",
                            },
                        },

                    }
                }
            }],
    )
    return response.choices[0].message

#使用大模型回复用户的问题，可能需要多次调用，如果需要使用工具，会返回使用什么工具，以及参数信息
async def get_llm_response(user_questions: str):
    messages = [
        {"role": "system", "content": "你是一个AI导游助手，可以查询景点和天气信息回答用户问题"},
        {"role": "user", "content": user_questions}
    ]
    response = await get_completion(messages)
    messages.append(response) #把大模型的回复加入到对话中

    #判断是否使用了工具
    while response.tool_calls is not None:
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)#将json字符串转换字典类型

            # 函数路由,判断使用哪个函数
            if tool_call.function.name == "get_areas_information":
                result = await get_areas_information(**args)
            elif tool_call.function.name == "get_weather_information":
                result = await get_weather_information(**args)

            #将使用的工具，以及使用工具返回的内容添加到信息中
            messages.append({
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)  # 数值result 必须转成字符串
            })
        #当工具使用完成后，继续调用大模型，如果得到最终答案，不需要使用工具，跳出循环，否则继续使用工具完成任务
        response = await get_completion(messages)
        messages.append(response)  # 把大模型的回复加入到对话中

    chat_messages = response.content
    return chat_messages

if __name__ == '__main__':
    user_questions = "碧桂园在哪里？"
    #user_questions = "可以看鸟的景点有哪些？"
    user_questions = "沈阳的天气怎么样？"
    response = asyncio.run(get_llm_response(user_questions))
    #response = await get_llm_response(user_questions)
    print(response)
