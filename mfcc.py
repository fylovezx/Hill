# 需要用到的包
import numpy, scipy, sklearn, librosa, librosa.display, matplotlib.pyplot as plt
# 录入咚哒咚咚哒的音频例子
x, fs = librosa.load('F:\\pythoncode\\data\\spe\\Recording.wav')
# 画出波形图（上方第一个图）
librosa.display.waveplot(x, sr=fs)
plt.show()
# 提取MFCC
mfccs = librosa.feature.mfcc(x, sr=fs)
# 获取特征值的维度
print(mfccs.shape)  #打印将输出(20,216)
# 画出MFCC的图（上方第二个图）
librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.show()
# 对MFCC的数据进行处理
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
# # 画出处理后的图（上方第三个图）
librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.show()