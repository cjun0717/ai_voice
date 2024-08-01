# main.py文件
开启websocket服务的函数，用于接收前端发送的语音数据，调用大模型进行回复，并将回复转换为语音并发送给前端

# openai_api.py文件
使用大模型的Function Calling功能回答用户问题的函数，包括工具函数的定义

# tts_utils.py文件
语音合成的函数，将文本转换为语音

# 运行
python main.py