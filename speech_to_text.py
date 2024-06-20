import pprint

import speech_recognition as sr

def speech_to_text(audio_name)->str:

    # 读取wav文件
    recognizer = sr.Recognizer()
    # 加载音频文件
    with sr.AudioFile(audio_name) as audio_file:
        # 读取音频数据
        audio_data = recognizer.record(audio_file)

        try:
            # 识别音频内容
            text = recognizer.recognize_google(audio_data, language="zh-CN")
            #print("Transcript: {}".format(text))
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == '__main__':
    test = speech_to_text("./output.wav")
    print(type(test))
    print(test)