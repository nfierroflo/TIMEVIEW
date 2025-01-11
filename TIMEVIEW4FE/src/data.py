import torch
from torch.utils.data import Dataset

def split(X_torch,Y_torch,train_ratio=0.7,val_ratio=0.15):
    n_samples = X_torch.size(0)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Shuffle the indices so we pick train/val/test randomly
    indices = torch.randperm(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Slice into train/val/test
    X_train, Y_train = X_torch[train_indices], Y_torch[train_indices]
    X_val,   Y_val   = X_torch[val_indices],   Y_torch[val_indices]
    X_test,  Y_test  = X_torch[test_indices],  Y_torch[test_indices]

    return X_train,Y_train,X_val,Y_val,X_test,Y_test

class TSDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]