import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy

from .data import split, TSDataset
from tqdm import tqdm
def train_tsmixer(
    model,
    X_torch,
    Y_torch,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=20,
    batch_size=32,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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
    #print(f"Best Model Test Loss: {test_loss:.4f}, Best Epoch: {best_epoch+1}")

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
    batch_size=32,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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
    #print(f"Best Model Test Loss: {test_loss:.4f}, Best Epoch: {best_epoch+1}")

    # Return the best model and the loss curves
    return model, (train_losses, val_losses), test_loss,test_dataset


def run_trials(
    train_func,
    model,
    X_torch,
    Y_torch,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_trials=5,
    seed=42,
    **kwargs
):
    """
    Run multiple trials of training the model and compute the average and standard deviation of test losses.

    Parameters
    ----------
    train_func : function
        The training function to use (either train_tsmixer or train_latent_tsmixer).
    model : nn.Module
        Your model instance.
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
    n_trials : int
        Number of trials to run.
    seed : int
        Seed for random number generation.
    kwargs : dict
        Additional keyword arguments for the training function.

    Returns
    -------
    best_model : nn.Module
        The best model (lowest validation loss) after training.
    (train_losses, val_losses) : tuple of lists
        The training and validation losses recorded at each epoch.
    test_loss_mean : float
        Mean of the test losses across trials.
    test_loss_std : float
        Standard deviation of the test losses across trials.
    """
    test_losses = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_trials)

    for trial in tqdm(range(n_trials)):
        torch.manual_seed(seeds[trial])
        np.random.seed(seeds[trial])

        # Call the training function
        _, _, test_loss, _ = train_func(
            model,
            X_torch,
            Y_torch,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            **kwargs
        )

        test_losses.append(test_loss)


    # Compute mean and standard deviation of test losses
    test_loss_mean = np.mean(test_losses)
    test_loss_std = np.std(test_losses)
    print(f"Mean Test Loss: {test_loss_mean:.4f}, Std Test Loss: {test_loss_std:.4f}")

    # Return the loss mean and std
    return test_loss_mean, test_loss_std