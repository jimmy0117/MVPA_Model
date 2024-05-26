# %%
import csv
import shutil
import pandas as pd
import tensorflow as tf
import librosa
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, LayerNormalization, Dropout ,Conv2D,Conv1D,Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

# %% 純DNN
def DNN_model_create(EHR):

    input1 = tf.keras.layers.Input(shape = EHR.shape)

    x1 = tf.keras.layers.Dense(16, activation='relu')(input1)

    # 添加其他層
    x = tf.keras.layers.Dense(32, activation='relu')(x1)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    output = tf.keras.layers.Dense(5, activation='softmax')(x)

    # 創建模型
    model = tf.keras.models.Model(inputs=input1, outputs=output)

    # 編譯模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 模型摘要
    model.summary()

    return model
# %% CNN + DNN整合模型
def model_create(EHR,mfcc):

    input1 = tf.keras.layers.Input(shape = EHR.shape)
    input2 = tf.keras.layers.Input(shape = mfcc.shape)

    # 假設這裡是您的模型結構
    # 可以是您自己定義的任何結構
    x1 = tf.keras.layers.Dense(16, activation='relu')(input1)
    x2 = tf.keras.layers.Conv1D(32, 2, activation='relu')(input2)
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    x2 = tf.keras.layers.Dense(32, activation='relu')(x2)
    # 合併兩個輸入
    x = tf.keras.layers.concatenate([x1, x2])

    # 添加其他層
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    output = tf.keras.layers.Dense(3, activation='softmax')(x)

    # 創建模型
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)

    # 編譯模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 模型摘要
    model.summary()

    return model


# %% 聲音特徵處理(取頻譜圖)
def process_audio(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = np.transpose(mel_spec)  # all frames
    if(len(mel_spec)==87):
        return mel_spec[3:83].transpose()
    elif(len(mel_spec)==130):
        return mel_spec[3+20:83+20].transpose()
    else:
        return -1

# %% 聲音特徵處理(取mfcc係數)
def audio_to_mfccs(filename, sample_rate=44100, offset=0, duration=None):
    voice, sample_rate = librosa.load(filename, sr=sample_rate, offset=offset, duration=duration)
    n_fft = int(16/1000 * sample_rate)  # Convert 16 ms to samples
    hop_length = int(8/1000 * sample_rate)  # Convert 8 ms to samples
    mfcc_feature = librosa.feature.mfcc(y=voice, sr=sample_rate, n_mfcc=128, n_fft=n_fft, hop_length=hop_length)
    
    # mfccs_cvn = (mfcc_feature - np.mean(mfcc_feature, axis=1, keepdims=True)) / np.std(mfcc_feature, axis=1, keepdims=True)
    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)
    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_features = np.transpose(mfccs)  # all frames
    
    if(len(mfccs_features)==376):
        return mfccs_features[95:95+200+1].transpose()
    elif(len(mfccs_features)==251):
        return mfccs_features[33:33+200+1].transpose()
    else:
        return -1

# merged_data.to_csv('./CNNtraning/merged_data.csv', index=False)  

# %% 處理病歷資料與聲音資料
fileName="Training_Dataset/training_datalist_resign.csv"
data=pd.read_csv(fileName)
EHR_x = data.copy()
mfccs_x = []
y = []
count = 0
for i in data["ID"]:
    audio_path = f"Training_Dataset/training_voice_data/{i}.wav"
    # mfccs_features = audio_to_mfccs(audio_path)
    mfccs_features = process_audio(audio_path)
    if type(mfccs_features) != int:
        
        mfccs_x.append(mfccs_features)
        y.append(data.loc[count, 'Disease category'])
    else:
        EHR_x = EHR_x.drop(count)
    count += 1

    # mfccs_y.append()


mfccs_x = np.array(mfccs_x)
y = np.array(y)
EHR_x = EHR_x.drop('ID', axis=1)
    
# %%

print(mfccs_x.shape)
# %%
# 標準化數據
#空值補零
EHR_x = EHR_x.fillna(0)
y = y - 1
numeric_columns = EHR_x.select_dtypes(include=[float, int]).columns
scaler = StandardScaler()
EHR_x[numeric_columns] = scaler.fit_transform(EHR_x[numeric_columns])
X = EHR_x[['Sex', 'Age','Narrow pitch range','Decreased volume','Fatigue','Dryness','Lumping','heartburn','Choking','Eye dryness','PND','Smoking','PPD','Drinking','frequency','Diurnal pattern','Onset of dysphonia ','Noise at work','Occupational vocal demand','Diabetes','Hypertension','CAD','Head and Neck Cancer','Head injury','CVA','Voice handicap index - 10']]  # 根據你的CSV文件中的特徵列名稱來提取特徵
y_categorical = to_categorical(y, num_classes=3)

EHR_X_train, EHR_X_test, EHR_y_train, EHR_y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
mfccs_X_train, mfccs_X_test, mfccs_y_train, mfccs_y_test = train_test_split(mfccs_x, y_categorical, test_size=0.2, random_state=42)
model = model_create(EHR_X_train.iloc[0],mfccs_x[0])


# %%
model.fit([EHR_X_train, mfccs_X_train], EHR_y_train, epochs=34, validation_data=([EHR_X_test, mfccs_X_test], EHR_y_test)) #34 is best


# %%
model = DNN_model_create(EHR_X_train.iloc[0])



# %%
model.fit(EHR_X_train, EHR_y_train, epochs=10, validation_data=(EHR_X_test, EHR_y_test))


# %%
# 預測測試集
predictions = model.predict([EHR_X_test, mfccs_X_test])

# 將預測的概率轉換為類別
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(EHR_y_test, axis=1)

# 計算混淆矩陣
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 顯示混淆矩陣
print("Confusion Matrix:")
print(conf_matrix)


# 產生混淆矩陣的圖表
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Class 2", "Class 1", "Class 0"])
disp.plot(cmap='Blues', values_format='d')

# 顯示圖表
plt.show()

# %%
# 繪製每個 epoch 的精確度折線圖
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()
# %%
model.save('voice_disease_model.h5')
# checkpoint = ModelCheckpoint('C:/Ubaya COVID-19 Voice Dataset/TRAIN_MODEL.h5', monitor='val_accuracy', save_best_only=True)
# %%
