import json
from time import perf_counter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import Tool
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.agents.react.agent import create_react_agent
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.agents.agent import AgentExecutor
from time import perf_counter
from dotenv import load_dotenv,find_dotenv
from langchain.agents.initialize import initialize_agent
import requests
import uvicorn
from fastapi import FastAPI, WebSocket
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv,find_dotenv
from prompt_utils import build_prompt
from faiss_utils import MyFaiss
from fastapi.middleware.cors import CORSMiddleware
from openai_utils import request
from speech_utils import text_to_speech
load_dotenv(find_dotenv())
app = FastAPI()
group_id = os.getenv("MINIMAX_GROUP_ID")
api_key = os.getenv("MINIMAX_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_tts_stream_headers() -> dict:
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + api_key,
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

# 初始化模型
llm = ChatOpenAI()

# 创建天气工具
tools = load_tools(["openweathermap-api"], llm)
agent_chain = initialize_agent(tools=tools, llm=llm, verbose=True)

# 创建一个检索问答工具链
vector_db = FAISS.load_local(
    folder_path="faiss_db",
    embeddings=OpenAIEmbeddings(),
    index_name="faiss_openai",
    allow_dangerous_deserialization=True,
)

RetrievalQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

tools = [
    Tool(
        name="AskDocument",
        func=RetrievalQA.run,
        description="根据用户的问题从文档中查询景点信息，回答用户问题"
    ),
    Tool(
        name="AskWeather",
        func=agent_chain.run,
        description="根据用户的问题查询天气信息，回答用户问题"
    )
]
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:

            user_quesation = await websocket.receive_text() #接收数据
            print(f'前端接收到的数据：{user_quesation}，开始调用大模型进行回复')

            llm_start_time = perf_counter()

            # print(tools)

            # 定义智能体提示词模板
            template1 = '''
            Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}
            '''


            prompt = PromptTemplate.from_template(template1)

            # 初始化Agent
            agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

            # 执行agent
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

            response = agent_executor.invoke({"input": user_quesation})
            chat_messages = response["output"]

            print(f"大模型回复：{chat_messages}")


            llm_end_time = perf_counter()
            print(f"智能体响应时间：{llm_end_time - llm_start_time}秒")

            # 将文本转换为语音
            print("开始将文本转换为语音")
            tts_url = "https://api.minimax.chat/v1/tts/stream?GroupId=" + group_id
            tts_headers = build_tts_stream_headers()
            #text = "真正的危险不是计算机开始像人一样思考，而是人开始像计算机一样思考。计算机只是可以帮我们处理一些简单事务。"
            tts_body = build_tts_stream_body(chat_messages)
            response = requests.request("POST", tts_url, stream=True, headers=tts_headers, data=tts_body)
            for chunk in (response.raw):
                if chunk:
                    if chunk[:5] == b'data:':
                        data = json.loads(chunk[5:])
                        if "data" in data and "extra_info" not in data:
                            if "audio" in data["data"]:
                                audio_hex = data["data"]['audio']
                                audio_bytes = bytes.fromhex(audio_hex)
                                await websocket.send_bytes(audio_bytes)


    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8765)

