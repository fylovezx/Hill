#引入库
import wave
import pyaudio

#定义数据流块
chunk = 1024

#只读方式打开wav文件
f = wave.open(r"F:\\pythoncode\\data\\spe\\Recording.wav","rb")

p = pyaudio.PyAudio()

#打开数据流
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
channels = f.getnchannels(),
rate = f.getframerate(),
output = True)

#读取数据
data = f.readframes(chunk)

#播放FileName.wav
while data != None:
	stream.write(data)
	data = f.readframes(chunk)

#停止数据流
stream.stop_stream()
stream.close()

#关闭 PyAudio
p.terminate()