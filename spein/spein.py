import pyaudio
import wave
#定义音频数据参数
CHUNK = 1024    #块
FORMAT = pyaudio.paInt16
CHANNELS = 2   #渠道
RATE = 44100   #率
RECORD_SECONDS = 5

WAVE_OUTPUT_FILENAME = "F:\\pythoncode\\data\\spe\\programisok.wav"

p = pyaudio.PyAudio()

# 打开数据流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("& Start Recording & :")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("#### done recording ####")

# 停止数据流  
stream.stop_stream()
stream.close()

# 关闭 PyAudio  
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()