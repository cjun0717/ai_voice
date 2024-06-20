from gtts import gTTS
import os

def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='zh-CN')
    tts.save(output_file)

if __name__ == '__main__':
    text_to_speech("你好，世界你好，世界，今天天气怎么样啊", "output.mp3")
    print("语音文件已保存为 output.mp3")


