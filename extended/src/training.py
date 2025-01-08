import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy

from .data import split, TSDataset

def train_tsmixer(
    model,
    X_torch,
    Y_torch,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=20,
    batch_size=16,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the TSMixer model on time-series data (X_torch, Y_torch),
    split into train/val/test in ratios (train_ratio, val_ratio, test_ratio).
    
    Parameters
    ----------
    model : nn.Module
        Your TSMixer model instance.
    X_torch : torch.Tensor
        Input tensor of shape (n_samples, input_channels, sequence_length).
    Y_torch : torch.Tensor
        Target tensor of shape (n_samples, input_channels, prediction_length).
    train_ratio : float
        Fraction of data to use for training.
    val_ratio : float
        Fraction of data to use for validation.
    test_ratio : float
        Fraction of data to use for testing.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for data loading.
    lr : float
        Learning rate for the optimizer.
    device : str
        Device to use for training ('cpu' or 'cuda').

    Returns
    -------
    best_model : nn.Module
        The best TSMixer model (lowest validation loss) after training.
    (train_losses, val_losses) : tuple of lists
        The training and validation losses recorded at each epoch.
    test_loss : float
        Final test set loss, computed using the best model state.
    """
    # ---------------------------------------------------------------------
    # 1. Split Data into Train/Val/Test
    # ---------------------------------------------------------------------
    
    X_train,Y_train,X_val,Y_val,X_test,Y_test = split(X_torch,Y_torch,train_ratio=train_ratio,val_ratio=val_ratio)

    # ---------------------------------------------------------------------
    # 2. Create PyTorch Datasets & DataLoaders
    # ---------------------------------------------------------------------
    train_dataset = TSDataset(X_train, Y_train)
    val_dataset   = TSDataset(X_val, Y_val)
    test_dataset  = TSDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ---------------------------------------------------------------------
    # 3. Set Up Model, Loss, Optimizer
    # ---------------------------------------------------------------------
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For tracking best model
    best_val_loss = float('inf')
    best_model_state = None

    # For storing loss curves
    train_losses = []
    val_losses = []

    # ---------------------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------------------
    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        total_train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward pass
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        # Average training loss
        epoch_train_loss = total_train_loss / len(train_loader.dataset)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                total_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = total_val_loss / len(val_loader.dataset)

        # Store losses for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Check if this is the best validation so far; if yes, save model state
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]: "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}")

    # ---------------------------------------------------------------------
    # 5. Load the Best Model State
    # ---------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ---------------------------------------------------------------------
    # 6. Final Evaluation on Test Set (using the best model)
    # ---------------------------------------------------------------------
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            total_test_loss += loss.item() * X_batch.size(0)

    test_loss = total_test_loss / len(test_loader.dataset)
    print(f"Best Model Test Loss: {test_loss:.4f}, Best Epoch: {best_epoch+1}")

    # Return the best model and the loss curves
    return model, (train_losses, val_losses), test_loss,test_dataset

def spline_coeff_loss(criterion, h, Y_batch, Phis):

    y_pred=[torch.matmul(torch.Tensor(Phi),h[d,:]) for d, Phi in enumerate(Phis[:h.shape[0]])]
    y_pred=torch.stack(y_pred)

    loss = criterion(y_pred, Y_batch)
    return loss

def train_latent_tsmixer(
    model,
    X_torch,
    Y_torch,
    Phis,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=20,
    batch_size=16,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the TSMixer model on time-series data (X_torch, Y_torch),
    split into train/val/test in ratios (train_ratio, val_ratio, test_ratio).
    
    Parameters
    ----------
    model : nn.Module
        Your TSMixer model instance.
    X_torch : torch.Tensor
        Input tensor of shape (n_samples, input_channels, sequence_length).
    Y_torch : torch.Tensor
        Target tensor of shape (n_samples, input_channels, prediction_length).
    train_ratio : float
        Fraction of data to use for training.
    val_ratio : float
        Fraction of data to use for validation.
    test_ratio : float
        Fraction of data to use for testing.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for data loading.
    lr : float
        Learning rate for the optimizer.
    device : str
        Device to use for training ('cpu' or 'cuda').

    Returns
    -------
    best_model : nn.Module
        The best TSMixer model (lowest validation loss) after training.
    (train_losses, val_losses) : tuple of lists
        The training and validation losses recorded at each epoch.
    test_loss : float
        Final test set loss, computed using the best model state.
    """
    # ---------------------------------------------------------------------
    # 1. Split Data into Train/Val/Test
    # ---------------------------------------------------------------------

    X_train,Y_train,X_val,Y_val,X_test,Y_test = split(X_torch,Y_torch,train_ratio=train_ratio,val_ratio=val_ratio)

    # ---------------------------------------------------------------------
    # 2. Create PyTorch Datasets & DataLoaders
    # ---------------------------------------------------------------------
    train_dataset = TSDataset(X_train, Y_train)
    val_dataset   = TSDataset(X_val, Y_val)
    test_dataset  = TSDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ---------------------------------------------------------------------
    # 3. Set Up Model, Loss, Optimizer
    # ---------------------------------------------------------------------
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For tracking best model
    best_val_loss = float('inf')
    best_model_state = None

    # For storing loss curves
    train_losses = []
    val_losses = []

    # ---------------------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------------------
    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        total_train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)


            # Forward pass
            h = model(X_batch)
            loss= spline_coeff_loss(criterion, h, Y_batch, Phis)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        # Average training loss
        epoch_train_loss = total_train_loss / len(train_loader.dataset)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                # Forward pass
                h = model(X_batch)
                loss= spline_coeff_loss(criterion, h, Y_batch, Phis)
                total_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = total_val_loss / len(val_loader.dataset)

        # Store losses for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Check if this is the best validation so far; if yes, save model state
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]: "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}")

    # ---------------------------------------------------------------------
    # 5. Load the Best Model State
    # ---------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ---------------------------------------------------------------------
    # 6. Final Evaluation on Test Set (using the best model)
    # ---------------------------------------------------------------------
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            h = model(X_batch)
            loss= spline_coeff_loss(criterion, h, Y_batch, Phis)
            total_test_loss += loss.item() * X_batch.size(0)

    test_loss = total_test_loss / len(test_loader.dataset)
    print(f"Best Model Test Loss: {test_loss:.4f}, Best Epoch: {best_epoch+1}")

    # Return the best model and the loss curves
    return model, (train_losses, val_losses), test_loss,test_dataset