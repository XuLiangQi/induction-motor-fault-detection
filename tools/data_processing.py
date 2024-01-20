import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def check_missing_data(data):
    if data.isnull().sum().sum() == 0:
        print('Checked, no missing value')
    else:
        print('missing value detected')
        data = data.drona()
    return data

def plot_FFT(data, rate):
    step = 5 * rate
    data = data.values
    x = len(data)/step
    for i in range(0, int(x)):
        i = step * i
        label = data[i, 8]
        dataFFT = fft(data[i:(step+i), 0:8], axis=0)
        dataFFT = dataFFT[1:, :]
        labels = (np.ones(len(dataFFT))*int(label)).reshape(len(dataFFT), 1)
        dataFFT = np.hstack((dataFFT, labels))
        if i == 0:
            totalFFT = dataFFT
        else:
            totalFFT = np.vstack((totalFFT, dataFFT))
        totalFFT = pd.DataFrame(abs(totalFFT))
    totalFFT.plot()
    plt.show()

def anomaly_detection(data):
    x = data.iloc[:, 0].values.reshape(len(data.values), 1)
    knn = NearestNeighbors(n_neighbors=5).fit(x)

    distances, _ = knn.kneighbors(x)
    maxHeight = max(distances.mean(axis=1))
    distances, maxHeight = anomaly_detection(data)
    plt.plot(pd.DataFrame(distances.mean(axis=1)))
    plt.axhspan(0.01, maxHeight, alpha=0.2, color='red')
    plt.show()

def resampling(data, rate):
  data = data.values
  x = len(data)/(50000 * 5)
  for i in range(0, int(x)):
    i = 250000 * i
    individualResampled = signal.resample(data[i : (250000 + i), :], rate) # resampled at 500 hz for each instance in 49 instances
    if i == 0:
      totalResampled = individualResampled
    else:
      totalResampled = np.vstack((totalResampled, individualResampled))
  return pd.DataFrame(totalResampled)


def create_lag(data):
  dataLabel = data[8]
  data = data.iloc[:, :8]
  dataShift = pd.concat([data.shift(4), data.shift(3), data.shift(2), data.shift(1), data, dataLabel], axis=1)
  dataShift = dataShift.dropna()
  return dataShift

def split_lable(data):
  dataLabel = data.iloc[:, 40]
  data = data.iloc[:, :40]
  return data.values, dataLabel.values

def standardize_data(train, test, val):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    val = scaler.transform(val)
    return train, test, val