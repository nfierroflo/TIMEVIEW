import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtsmixer import TSMixer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.training import train_latent_tsmixer_seasonality,run_trials

from timeview.basis import BSplineBasis
from timeview.knot_selection import calculate_knot_placement

#add a path to the system path
import sys
sys.path.append('../')
from experiments.datasets import *

seed=42
torch.manual_seed(seed)

n_trials=10

sequence_length = 1
prediction_length = 20
input_channels = 1
output_channels = 2
n_samples=200
n_timesteps=prediction_length

results=pd.DataFrame(columns=['dataset','model','mean_loss','std_loss'])

print("Seasonality experiments")
print("Seasonality1")

X = pd.DataFrame({'x':np.linspace(1.0,3.0,n_samples)})
ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
ys = [2*t*x + np.sin(t*x*np.pi) for t, x in zip(ts, X['x'])]


Ys=np.array(ys)
X_torch=torch.tensor(X.values).float().reshape(n_samples,sequence_length,input_channels)
Y_torch=torch.tensor(Ys).float().reshape(n_samples,prediction_length,1)




B=6
t=ts[0]

internal_knots=calculate_knot_placement(ts, ys, n_internal_knots=B-2, T=t[-1],seed=0, verbose=False)

bspline=BSplineBasis(n_basis=B,t_range=(t[0],t[-1]),internal_knots=internal_knots)
Phis = list(bspline.get_all_matrices(np.array(ts)))


latent_model = TSMixer(
    sequence_length=sequence_length,   # same as time steps in X
    prediction_length=B, # number of spline coefficients to predict
    input_channels=1,
    output_channels=output_channels
)

best_latent_model, curves, test_loss, test_dataset = train_latent_tsmixer_seasonality(
    latent_model,
    X_torch,
    Y_torch,
    Phis,
    ts,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=400,
    batch_size=32,
    lr=1e-2,
    device="cpu"  # or "cuda"
)

test_loss_mean, test_loss_std= run_trials(
    train_latent_tsmixer_seasonality,
    latent_model,
    X_torch,
    Y_torch,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=400,
    batch_size=32,
    lr=1e-2,
    device= "cpu",
    n_trials=n_trials,
    seed=42,
    Phis=Phis,
    ts=ts
    )

new_row = pd.DataFrame({'dataset': ['Seasonality1'], 'model': ['TIMEVIEW4FE(TSMixer)'], 'mean_loss': [test_loss_mean], 'std_loss': [test_loss_std]})
results = pd.concat([results, new_row], ignore_index=True)

print("Seasonality2")

X = pd.DataFrame({'x':np.linspace(1.0,3.0,n_samples)})
ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
ys = [2*t*x + x*np.sin(t*x*np.pi+np.exp(-x)) for t, x in zip(ts, X['x'])]

Ys=np.array(ys)
X_torch=torch.tensor(X.values).float().reshape(n_samples,sequence_length,input_channels)
Y_torch=torch.tensor(Ys).float().reshape(n_samples,prediction_length,1)


B=6
t=ts[0]

internal_knots=calculate_knot_placement(ts, ys, n_internal_knots=B-2, T=t[-1],seed=0, verbose=False)

bspline=BSplineBasis(n_basis=B,t_range=(t[0],t[-1]),internal_knots=internal_knots)
Phis = list(bspline.get_all_matrices(np.array(ts)))


latent_model = TSMixer(
    sequence_length=sequence_length,   # same as time steps in X
    prediction_length=B, # number of spline coefficients to predict
    input_channels=1,
    output_channels=output_channels
)

best_latent_model, curves, test_loss, test_dataset = train_latent_tsmixer_seasonality(
    latent_model,
    X_torch,
    Y_torch,
    Phis,
    ts,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=400,
    batch_size=32,
    lr=1e-2,
    device="cpu"  # or "cuda"
)

test_loss_mean, test_loss_std= run_trials(
    train_latent_tsmixer_seasonality,
    latent_model,
    X_torch,
    Y_torch,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=20,
    batch_size=32,
    lr=1e-2,
    device= "cpu",
    n_trials=n_trials,
    seed=42,
    Phis=Phis,
    ts=ts
    )

new_row = pd.DataFrame({'dataset': ['Seasonality2'], 'model': ['TIMEVIEW4FE(TSMixer)'], 'mean_loss': [test_loss_mean], 'std_loss': [test_loss_std]})
results = pd.concat([results, new_row], ignore_index=True)

results.to_csv('results/Seasonality_results.csv', mode='a', header=False)