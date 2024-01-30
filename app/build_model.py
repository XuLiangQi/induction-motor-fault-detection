import pandas as pd
from app.tools.data_processing import (PTDataset)
import torch
from torch.utils.data import DataLoader
from app.model.model import FeedforwardModel, train, calc_metrics
from hyps.hyps import Hyperparameters


def build_model(X_train: pd.DataFrame
                , y_train: pd.DataFrame
                , X_val: pd.DataFrame
                , y_val: pd.DataFrame 
                , X_test: pd.DataFrame 
                , y_test: pd.DataFrame ) -> None:

    device = ""
    if torch.cuda.is_available():
        device = "cuda"
        print(" CUDA available, will be training the model on CUDA GPU. ")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(" MPS backend available, will be training the model on MAC GPU")
    else:
        device = "cpu"
        print(" No GPU found, will be training the model on CPU. ")
    device = torch.device(device)
    model = FeedforwardModel(len(X_train.T), len(y_train.T), Hyperparameters.dropout_fraction, device)
    model.to(device)
    train_dataset = PTDataset(X_train, y_train)
    val_dataset = PTDataset(X_val, y_val)
    test_dataset = PTDataset(X_test, y_test)

    train(model, train_dataset, val_dataset, Hyperparameters.max_iters
          , Hyperparameters.batch_size, Hyperparameters.lr, Hyperparameters.early_stop, device)

    model.eval()
    data_loader_val = DataLoader(
        test_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=32)
    loss_test, accuracy_test = calc_metrics(model, data_loader_val, device)
    print('Final model performance metrics:')
    print('\tLoss (test): {0}\tAccuracy (test): {1}'.format(
        round(loss_test, 4), round(accuracy_test, 4))
    )