from gtts import gTTS
import os
import platform

text = "Raise up your head!"

# 判断系统平台
plat = platform.system()

# 文本转语音 
tts = gTTS(text=text, lang='en')  
tts.save("test.mp3")

if plat == 'Darwin': # macOS
    # Mac语音合成
    os.system("say " + text) 