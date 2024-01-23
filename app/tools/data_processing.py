import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from app.tools.data_loader import load_data
import torch
from torch.utils.data import Dataset

def downsample_data(data, rate):
    downsampled_data = pd.DataFrame()
    selection_start = 0
    selection_end = rate
    for rows in range(int(len(data)/rate)):
        selected_rows = data.iloc[selection_start : selection_end, :]
        avg_selection = selected_rows.sum()/rate;
        avg_selection = pd.DataFrame(avg_selection.values.reshape(1, len(avg_selection)))
        downsampled_data = pd.concat([downsampled_data, avg_selection], ignore_index=True, axis=0)
        selection_start += rate
        selection_end = selection_start + rate
    return downsampled_data

def FFT(data):
    autocorr = signal.fftconvolve(data,data[::-1],mode='full')
    return pd.DataFrame(autocorr)


def standardize_data(train, test, val):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    val = scaler.transform(val)
    return train, test, val

def one_hot_encoding(y_train, y_test, y_val):
   encoder = OneHotEncoder()
   encoder.fit(y_train)
   y_train = encoder.transform(y_train).toarray()
   y_test = encoder.transform(y_test).toarray()
   y_val = encoder.transform(y_val).toarray()
   return y_train, y_test, y_val

class PTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        length = len(self.data)
        return length
    
    def __getitem__(self, index):
        device = ""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        data_point = torch.tensor(self.data.iloc[index, :]).float()
        data_point.to(torch.device(device))
        label = torch.tensor(self.labels[index, :]).float()
        label.to(torch.device(device))

        return data_point, label
    

def pre_process_data():
    print("Start loading data ...")
    data_n, data_6g, data_10g, data_15g, data_20g, data_25g, data_30g, data_35g = load_data()
    print("Data successfully loaded.")

    print("Downsamping data ... ")
    data_n = downsample_data(data_n, 5000)
    data_6g = downsample_data(data_6g, 5000)
    data_10g = downsample_data(data_10g, 5000)
    data_15g = downsample_data(data_15g, 5000)
    data_20g = downsample_data(data_20g, 5000)
    data_25g = downsample_data(data_25g, 5000)
    data_30g = downsample_data(data_30g, 5000)
    data_35g = downsample_data(data_35g, 5000)
    print("Data downsampled on a rate of ", 5000)

    print("FFT converting data into frequency domain ... ")
    data_n = FFT(data_n)
    data_6g = FFT(data_6g)
    data_10g = FFT(data_10g)
    data_15g = FFT(data_15g)
    data_20g = FFT(data_20g)
    data_25g = FFT(data_25g)
    data_30g = FFT(data_30g)
    data_35g = FFT(data_35g)
    print("FFT completed, data is now in frequency domain")

    data = pd.concat([data_n,data_6g,data_10g,data_15g,data_20g,data_25g,data_30g,data_35g],ignore_index=True, axis=0)
    y_0 = pd.DataFrame(np.ones(int(len(data_n)),dtype=int))
    y_1 = pd.DataFrame(np.zeros(int(len(data_6g)),dtype=int))
    y_2 = pd.DataFrame(np.full((int(len(data_10g)),1),2))
    y_3 = pd.DataFrame(np.full((int(len(data_15g)),1),3))
    y_4 = pd.DataFrame(np.full((int(len(data_20g)),1),4))
    y_5 = pd.DataFrame(np.full((int(len(data_25g)),1),5))
    y_6 = pd.DataFrame(np.full((int(len(data_30g)),1),6))
    y_7 = pd.DataFrame(np.full((int(len(data_35g)),1),7))
    labels = pd.concat([y_0, y_1,y_2,y_3,y_4,y_5,y_6,y_7], ignore_index=True, axis=0)

    print("Splitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    print(f"Data splited. Train data: {X_train.shape}, Validation data: {X_val.shape}, Test data: {X_test.shape}")

    y_train, y_test,  y_val = one_hot_encoding(y_train, y_test,  y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test