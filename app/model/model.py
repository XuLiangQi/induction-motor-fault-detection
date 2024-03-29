import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

class FeedforwardModel(nn.Module):
    '''A vanilla neural network model used to predict the type
    of the motor failure. 
    
    Attributes:
    ----------
    device: str:
        The device of which the model will be trained on.

    fc: Sequential
        The fully connected linear layers used to predict the type of motor failure.
    '''
    def __init__(self, 
                 num_features: int, 
                 num_classes: int,
                 dropout_fraction: float,
                 device):
        super(FeedforwardModel, self).__init__()
        self.device = device
        self.fc = nn.Sequential(nn.Linear(num_features, 32),
                                nn.ReLU(),
                                nn.Linear(32, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, num_classes)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, a=0, b=1)

    def forward(self, X_train):
        X_train = X_train.to(self.fc[0].weight.dtype)
        X_train = X_train.to(self.device)
        out = self.fc(X_train)
        return out

def calc_metrics(model: nn.Module,
                 data_loader: DataLoader,
                 device: str) -> (float, float):
    '''Quantify the performance of the model during training
    phase.
    
    Calculate the loss and accuracy of the model's prediction during
    each batch on each epoch. Calculates the average of all the batches
    to obtain more repersentative performance metrics.
    
    Parameters:
    ----------

    model: nn.Module
        The model that needs to be evaluated.
        
    data_loader: DataLoader
        The pytorch data loader object to decompose the data into batches.
        
    device: str
        The device of which the model will be trained on.
        
    Returns:
    -------
    
    loss_mean: float
        The average loss calculated from all batches.
        
    accuracy_mean: float
        The average accuracy calculated from all batches.
    '''
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    with torch.no_grad():
        for xb, yb in data_loader:       
            xb = xb.to(device)
            yb = yb.to(device)
            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            losses.append(loss.item())
            yb_p = F.softmax(yb_p, dim=1)
            yb_p = torch.round(yb_p)
            for i in range(len(yb_p)):
                check_array = yb_p[i] == yb[i]
                if not check_array.all():
                    accuracies.append(torch.tensor(0))
                else:
                    accuracies.append(torch.tensor(1))

    loss_mean = sum(losses)/float(len(losses))
    accuracy_mean = sum(accuracies)/float(len(accuracies))
    return loss_mean, accuracy_mean.item()

def train(model: nn.Module,
        dataset_train: Dataset,
        dataset_val: Dataset,
        max_epochs: int,
        batch_size: int,
        lr: float,
        early_stop: int,
        device: str):
    '''Train the model based on parameters provided.
    
    Parameters:
    ---------
    
    model: nn.Module
        The model that needs to be trained.
        
    dataset_train: Dataset
        A custom dataset used to feed into dataloader to split
        training data into batches.
        
    dataset_val: Dataset
        A custom dataset used to feed into dataloader to split
        validation data into batches.
        
    max_epochs: int
        The maximum epoches (iterations) that the training will perform.
        
    batch_size: int
        The number of data points (rows) contained in each batch.
        
    lr: float
        The learning rate used by the optimizer.
        
    early_stop: int
        The number of epochs to stop the training earlier
        if the loss is not impoving.
        
    device: str
        The device of which the model will be trained on.
    '''
    model.train()
    print("Model is running on:", next(model.parameters()).device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    data_loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size)
    
    data_loader_val = DataLoader(
        dataset_val,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size)
    
    loss_val_min = float('inf')
    best_model = {}
    n_epochs_since_best = 0
    for epochs in range(max_epochs):
        for xb, yb in data_loader_train:            
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        loss_train, accuracy_train = calc_metrics(
            model, data_loader_train, device)
        loss_val, accuracy_val = calc_metrics(
            model, data_loader_val, device)
        model.train()

        new_best = False
        if loss_val < loss_val_min:
            loss_val_min = loss_val
            best_model = model.state_dict()
            new_best = True
            n_epochs_since_best = 0
        else:
            n_epochs_since_best += 1

        output_msg = (
            '\tEpoch: {}\tLoss (train): {:.4f}\tLoss (val): {:.4f}\t'
            'Accuracy (train): {:.4f}\tAccuracy (val): {:.4f}'.
            format(
                epochs,
                round(loss_train, 4), 
                round(loss_val, 4), 
                round(accuracy_train, 4),
                round(accuracy_val, 4)))
        if new_best:
            output_msg += '\t*'
        elif n_epochs_since_best >= early_stop:
            # TODO: Replace with propoer detect_overfitting() function
            # that takes in the history of validation losses
            print('\tOverfitting detected - training terminated')
            break
        print(output_msg)

    model.load_state_dict(best_model)
    model.eval()
    