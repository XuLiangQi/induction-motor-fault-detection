import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.data_processing import (check_missing_data, plot_FFT
                                   , anomaly_detection, resampling
                                   , create_lag, split_lable, standardize_data)
from model.LSTM import LSTMModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataTrain = pd.read_csv('data/reconstituted_data/data_train.csv')
dataTest = pd.read_csv('data/reconstituted_data/data_test.csv')
dataVal = pd.read_csv('data/reconstituted_data/data_val.csv')
print("Data loaded successfully")

# data = check_missing_data(dataVal)
# data.iloc[:250000, :].plot()
# plt.show()

#FFT(data.iloc[:250000, :], 50000)

#distances, maxHeight = anomaly_detection(data)

dataTrain = resampling(dataTrain, 500)
dataTest = resampling(dataTest, 500)
dataVal = resampling(dataVal, 500)
print("Data resampled")
#FFT(dataValResampled.iloc[:500, :], 100)

dataTrain = create_lag(dataTrain)
dataTest = create_lag(dataTest)
dataVal = create_lag(dataVal)
print("Lag created")

X_train, y_train = split_lable(dataTrain)
X_test, y_test = split_lable(dataTest)
X_val, y_val = split_lable(dataVal)

X_train, X_test, X_val = standardize_data(X_train, X_test, X_val)
print("Data standardized")


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

model = LSTMModel(X_train, y_train, X_test, y_test)
model.train()
#model.plot()
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)


print("pred:", y_pred.shape)
print("real:", y_val.shape)

y_val = np.round(y_val).astype(int)
y_pred = np.round(y_pred).astype(int)

accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")