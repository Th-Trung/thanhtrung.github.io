from statistics import mode
import numpy as np
import pandas as pd
import keras
import sklearn
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
# Đọc dữ liệu
handswing_df = pd.read_csv("Handswing.txt")
Body_df = pd.read_csv("Body.txt")
X=[]
y=[]
no_of_timesteps =10
dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])# ghép 10 timestep
    y.append(1) # 1: lớp Handswing

dataset = Body_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])# ghép 10 timestep
    y.append(0) # 0: lớp Body

X,y = np.array(X), np.array(y)
print(X.shape,y.shape)

# Chia bộ dự liệu để test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Xây dựng model LSTM Train dữ liệu

model = Sequential()
model.add(LSTM(units = 128, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units= 1, activation = "sigmoid"))# Sigmoid hàm kích hoạt
model.compile(optimizer = "adam", metrics = ['accuracy'], loss= "binary_crossentropy")
model.fit(X_train, y_train, epochs = 50, batch_size = 32, validation_data = (X_test, y_test))
model.save("Train_model.h5")
