import time
import datetime
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

    # 4) 전처리 데이터 npy저장 및 로드
# np.save('./_save/_npy/project/soccer_prediction/SP_x.npy', arr=x)
# np.save('./_save/_npy/project/soccer_prediction/SP_y.npy', arr=y)
# np.save('./_save/_npy/project/soccer_prediction/SP_x_pred.npy', arr=x_pred)
x = np.load('./_save/_npy/project/soccer_prediction/SP_x.npy')
y = np.load('./_save/_npy/project/soccer_prediction/SP_y.npy')
x_pred = np.load('./_save/_npy/project/soccer_prediction/SP_x_pred.npy')

    # 5) train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=21)

#2. Modeling
input = Input((40,))
d = Dense(512, activation='relu')(input)
d = Dense(256, activation='relu')(d)
output = Dense(1, activation='relu')(d)
model = Model(inputs=input, outputs=output)
# model = RandomForestRegressor() 

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/_mcp/project/soccer_prediction/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'SP', '_', date_time, '_', info, '.hdf5'])
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=10)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.1, callbacks=[es, cp])
# model.fit(x_train, y_train) # RandomForestRegressor
end_time = time.time() - start_time

#4. Evaluating, Prediction
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
prediction = model.predict(x_pred)

print('loss = ', loss)
print('r2 score =', r2)
print('Performance Prediction = ', prediction)

'''
loss =  0.9929283857345581
r2 score = 0.9737266283123291
Performance Prediction =  [[84.824684]]

RandomForestRegressor
r2 score = 0.9451363556335565
Performance Prediction =  [82.08301616]
'''
