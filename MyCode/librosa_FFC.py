import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Training_Dataset/training_voice_data/0701355.wav'
file_path = 'Training_Dataset/training_voice_data/1201pl2.wav'
y, sr = librosa.load(file_path, sr=None)

print(len(y))
plt.figure(figsize=(100, 3)) 

librosa.display.waveshow(y, sr=sr)
plt.title('Normal')
plt.show()
