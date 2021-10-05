import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#1. data preprocessing
datasets = load_iris()
x = datasets.data
y = datasets.target
scaler = StandardScaler()
scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)
parameter = [
    {'C': [1,10,100,1000], 'kernel': ['linear']},
    {'C': [1,10,100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1,10,100,1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
]

model = SVC()
grid = GridSearchCV(model, parameter, cv=kfold)
grid.fit(x_train, y_train)

best_estimator = grid.best_estimator_
best_score = grid.best_score_
y_pred = grid.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
grid_score = grid.score(x_test, y_test)

print('best parameter = ', best_estimator)
print('best score = ', best_score)
print('acc score = ', acc_score)
print('grid score = ', grid_score)

# best parameter =  SVC(C=1, kernel='linear')
# best score =  0.9666666666666668
# acc score =  0.9666666666666667
# grid score =  0.9666666666666667

model = SVC(C=1, kernel='linear') # gridsearch 결과 확인
model.fit(x_train, y_train)
model_score = model.score(x_test, y_test)

print('model score = ', model_score)
# model score =  0.9666666666666667
