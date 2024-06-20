from typing import List, Dict
from dotenv import load_dotenv,find_dotenv
from openai import OpenAI
_ = load_dotenv(find_dotenv())
client = OpenAI()
def get_llm_response(messages: List[Dict[str, str]]):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    response_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content

    return response_text

if __name__ == '__main__':
    text = get_llm_response([{"role": "user", "content": "你好"}])
    print(text)