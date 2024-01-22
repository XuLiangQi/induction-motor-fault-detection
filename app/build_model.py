from app.tools.data_processing import (PTDataset)
import torch
from torch.utils.data import DataLoader
from app.model.model import FeedforwardModel, train, calc_metrics


def build_model(X_train, y_train, X_val, y_val, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedforwardModel(len(X_train.T), 8, 0.2, device)
    model.to(device)
    train_dataset = PTDataset(X_train, y_train)
    val_dataset = PTDataset(X_val, y_val)
    test_dataset = PTDataset(X_test, y_test)

    train(model, train_dataset, val_dataset, 100, 32, 0.01, device)

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