import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from app.tools.data_loader import load_data
import torch
from torch.utils.data import Dataset

def downsample_data(data: pd.DataFrame
                    , rate: int) -> None:
    '''Reduce the size of the data based on provided rate.
    Downsampling data to an appropriate size is important to improve the
    model's training speed, as larger the dataset will result in longer 
    training time. 
    
    Parameters:
    ----------
    
    data: pd.DataFrame
        The data that needs to be downsampled
        
    rate: int
        The rate for downsampling. For example, a rate of 100 means
        1 data point (row) will be selected every 100 data points (rows).
        For a dataset of 1000 rows the result will ended up 10 rows after
        downsampling.
    '''
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

def FFT(data: pd.DataFrame) -> pd.DataFrame:
    '''Calculates the correlation matrix in the data 
    by using FFTconvolve method.
    
    Parameters:
    ----------
    
    data: pd.DataFrame
        The data of which the correlation calculation is needed.
        
    Returns:
    -------

    autocorr: pd.DataFrame
        The correlation matrix.
    '''
    autocorr = signal.fftconvolve(data,data[::-1],mode='full')
    return pd.DataFrame(autocorr)


def standardize_data(train: pd.DataFrame
                     , test: pd.DataFrame
                     , val: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Standarize the data.
    
    Parameters:
    ----------
    
    train: pd.DataFrame
        The training dataset.
    
    test: pd.DataFrame
        The testing dataset.
        
    val: pd.DataFrame
        The validation dataset.
        
    Returns:
    -------

    train: pd.DataFrame
        The standardized training dataset.
    
    test: pd.DataFrame
        The standardized testing dataset.
        
    val: pd.DataFrame
        The standardized validation dataset.
        
    '''
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    val = scaler.transform(val)
    return pd.DataFrame(train), pd.DataFrame(test), pd.DataFrame(val)

def one_hot_encoding(y_train: pd.DataFrame
                     , y_test: pd.DataFrame
                     , y_val: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
   '''One hot encode the categorical data
   
   Parameters:
   ---------

   y_train: pd.DataFrame
       The training label.
   
   y_test: pd.DataFrame
       The testing label.
       
   y_val: pd.DataFrame
       The validation label.
   
   Returns:
   -------
       
   y_train: pd.DataFrame
       The one hot encoded training label.
   
   y_test: pd.DataFrame
       The one hot encoded testing label.
       
   y_val: pd.DataFrame
       The one hot encoded validation label.
   '''
   encoder = OneHotEncoder()
   encoder.fit(y_train)
   y_train = encoder.transform(y_train).toarray()
   y_test = encoder.transform(y_test).toarray()
   y_val = encoder.transform(y_val).toarray()
   return y_train, y_test, y_val

class PTDataset(Dataset):
    '''Pytorch custom dataset is used to store both data and label.
    This class will make batch process easier in the model training phase. 
    
    Attributes:
    ----------
    
    data: pd.DataFrame
        The data without the label in form of DataFrame

    labels: pd.DataFrame
        The label in form of DataFrame

    device: str
        The device of which the model will be trained on.
    '''
    def __init__(self, data: pd.DataFrame
                 , labels: pd.DataFrame
                 , device: str):
        self.data = data
        self.labels = labels
        self.device = device

    def __len__(self):
        length = len(self.data)
        return length
    
    def __getitem__(self, index):
        data_point = torch.tensor(self.data.iloc[index, :]).float()
        data_point.to(torch.device(self.device))
        label = torch.tensor(self.labels[index, :]).float()
        label.to(torch.device(self.device))

        return data_point, label
    

def pre_process_data() -> (pd.DataFrame, pd.DataFrame
                           , pd.DataFrame, pd.DataFrame
                           , pd.DataFrame, pd.DataFrame):
    '''Preprocess the data before feeding into the actual model.
    The preprocessing package include downsampling the dataset, using 
    FFTconvolve to reveal the correlation between features, and splitting 
    data into training, validating, and testing sets. 

    Returns:
    -------

    X_train: pd.DataFrame
        The training data.
    
    y_train: pd.DataFrame
        The training label.
    
    X_val: pd.DataFrame 
        The validation data, used for validation during model training.
    
    y_val: pd.DataFrame
        The validation label, used for validation during model training.

    X_test: pd.DataFrame 
        The testing data, used for testing the final model after trained.

    y_test: pd.DataFrame
        The testing label, used for testing the final model after trained.

    '''
    print("Start loading data ...")
    data_n, data_6g, data_10g, data_15g, data_20g, data_25g, data_30g = load_data()
    print("Data successfully loaded.")

    print("Downsamping data ... ")
    data_n = downsample_data(data_n, 5000)
    data_6g = downsample_data(data_6g, 5000)
    data_10g = downsample_data(data_10g, 5000)
    data_15g = downsample_data(data_15g, 5000)
    data_20g = downsample_data(data_20g, 5000)
    data_25g = downsample_data(data_25g, 5000)
    data_30g = downsample_data(data_30g, 5000)
    print("Data downsampled on a rate of ", 5000)

    print("FFT converting data into frequency domain ... ")
    data_n = FFT(data_n)
    data_6g = FFT(data_6g)
    data_10g = FFT(data_10g)
    data_15g = FFT(data_15g)
    data_20g = FFT(data_20g)
    data_25g = FFT(data_25g)
    data_30g = FFT(data_30g)
    print("FFT completed, data is now in frequency domain")

    data = pd.concat([data_n,data_6g,data_10g,data_15g,data_20g,data_25g,data_30g],ignore_index=True, axis=0)
    y_0 = pd.DataFrame(np.ones(int(len(data_n)),dtype=int))
    y_1 = pd.DataFrame(np.zeros(int(len(data_6g)),dtype=int))
    y_2 = pd.DataFrame(np.full((int(len(data_10g)),1),2))
    y_3 = pd.DataFrame(np.full((int(len(data_15g)),1),3))
    y_4 = pd.DataFrame(np.full((int(len(data_20g)),1),4))
    y_5 = pd.DataFrame(np.full((int(len(data_25g)),1),5))
    y_6 = pd.DataFrame(np.full((int(len(data_30g)),1),6))
    labels = pd.concat([y_0, y_1,y_2,y_3,y_4,y_5,y_6], ignore_index=True, axis=0)

    print("Splitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    print(f"Data splited. Train data: {X_train.shape}, Validation data: {X_val.shape}, Test data: {X_test.shape}")

    X_train, X_test, X_val = standardize_data(X_train, X_test, X_val)

    y_train, y_test,  y_val = one_hot_encoding(y_train, y_test,  y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test