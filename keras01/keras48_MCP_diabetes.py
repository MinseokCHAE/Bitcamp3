import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# print(x_train.shape, x_test.shape) # (353, 10, 1) (89, 10, 1)
# print(y_train.shape, y_test.shape) # (353,) (89,)

# #2. modeling
# input = Input(shape=(10, 1))
# x = Conv1D(64, (2,), activation='relu')(input)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = MaxPooling1D()(x)
# x = Conv1D(128, (1,), activation='relu')(x)
# x = Conv1D(256, (1,), activation='relu')(x)
# x= Dropout(0.2)(x)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = Conv1D(32, (2,), activation='relu')(x)
# x = GlobalAveragePooling1D()(x)
# output = Dense(1, activation='relu')(x)

# model = Model(inputs=input, outputs=output)

#3. compiling, training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                                         filepath='./_save/MCP/keras48_MCP_diabetes.hdf5')
# es = EarlyStopping(monitor='val_loss', patience=32, mode='min', verbose=1)
# model.compile(loss='mse', optimizer='adam')
start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=128, batch_size=16, 
#                             validation_split=0.05, callbacks=[es, cp])
end_time = time.time() - start_time

model = load_model('./_save/MCP/keras48_MCP_diabetes.hdf5')

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print('loss = ', loss)
print('R2 score = ', r2)
print('time taken(s) : ', end_time)

'''
loss =  4303.9296875
R2 score =  0.3368409755795422
time taken(s) :  7.813257694244385

load_MCP
loss =  5787.0205078125
R2 score =  0.10832292710335001
'''
