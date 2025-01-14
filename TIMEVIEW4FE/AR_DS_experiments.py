import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtsmixer import TSMixer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeview.basis import BSplineBasis
from timeview.knot_selection import calculate_knot_placement
from src.training import train_tsmixer,train_latent_tsmixer,run_trials
from timeview.basis import BSplineBasis
from timeview.knot_selection import calculate_knot_placement
import warnings
warnings.filterwarnings("ignore")
#add a path to the system path
import sys
sys.path.append('../')

from experiments.datasets import *
seed=42
torch.manual_seed(seed)


def execute_trials(dataset_params,dataset_name,n_samples, n_timesteps, B, sequence_length, prediction_length, input_channels, output_channels, epochs, n_trials,fixed_len=None,T=None,Ti=None):
    #call the dataset generator
    Dataset_generator=eval(dataset_name)(**dataset_params)
    print(f"{dataset_name}")
    #generate a dataset
    if fixed_len is not None:
        X,ts,ys=Dataset_generator.get_X_ts_ys(fixed_len)
    else:
        X,ts,ys=Dataset_generator.get_X_ts_ys()

    Ys=np.array(ys)

    X_torch=torch.tensor(Ys[:,:sequence_length]).float().reshape(n_samples,sequence_length,1)
    Y_torch=torch.tensor(Ys[:,sequence_length:]).float().reshape(n_samples,prediction_length,1)
    
    print("TSMixer")
    model= TSMixer(
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        input_channels=input_channels,
        output_channels=output_channels,
    )


    mean_loss,std_loss=run_trials(
        train_tsmixer,
        model,
        X_torch,
        Y_torch,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        epochs=epochs,
        batch_size=32,
        lr=1e-2,
        device="cpu",
        n_trials=n_trials,
        seed=42,
    )



    ts_N=[arr[sequence_length:] for arr in ts]
    ys_N=[arr[sequence_length:] for arr in ys]

    if T is None:
        T=ts_N[0][-1]
    if Ti is None:
        Ti=ts_N[0][0]
    internal_knots=calculate_knot_placement(ts_N, ys_N, n_internal_knots=B-2, T=T, Ti=Ti ,seed=0, verbose=False)

    bspline=BSplineBasis(n_basis=B,t_range=(Ti,T),internal_knots=internal_knots)
    Phis = list(bspline.get_all_matrices(np.array(ts)[:,sequence_length:]))

    latent_model = TSMixer(
        sequence_length=sequence_length,   # same as time steps in X
        prediction_length=B, # number of spline coefficients to predict
        input_channels=input_channels,
        output_channels=output_channels,
    )


    print("TSMixer as Encoder")
    mean_loss,std_loss=run_trials(
        train_latent_tsmixer,
        latent_model,
        X_torch,
        Y_torch,
        Phis=Phis,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        epochs=200,
        batch_size=32,
        lr=1e-2,
        device="cpu",
        n_trials=10,
        seed=42,
    )

#SineTransDataset_AR
dataset_dict={'n_samples':200,'n_timesteps':60}
trial_dict={'dataset_name':"SineTransDataset_AR",'dataset_params':dataset_dict,'sequence_length':40,'prediction_length':20,'input_channels':1,'output_channels':1,'n_samples':200,'n_timesteps':60, 'B':5,'epochs':200,'n_trials':10}

execute_trials(**trial_dict)

sequence_length = 40
prediction_length = 20
input_channels = 1
output_channels = 1
n_samples=900

#BetaDataset
dataset_dict={'n_samples':900,'n_timesteps':60}
trial_dict={'dataset_name':"BetaDataset",'dataset_params':dataset_dict,'sequence_length':40,'prediction_length':20,'input_channels':1,'output_channels':1,'n_samples':900,'n_timesteps':60, 'B':5,'epochs':200,'n_trials':10}

execute_trials(**trial_dict)

#SyntheticTumorDataset
dataset_dict={
'n_samples': 2000,'n_time_steps': 60,'time_horizon': 1.0,'noise_std': 0.0,'seed': 0,'equation': 'wilkerson'
}
trial_dict={'dataset_name':"SyntheticTumorDataset",'dataset_params':dataset_dict,'sequence_length':40,'prediction_length':20,'input_channels':1,'output_channels':1,'n_samples':2000,'n_timesteps':60, 'B':9,'epochs':200,'n_trials':10}

execute_trials(**trial_dict)

os.chdir('../experiments')
#FLChainDataset
dataset_dict={}
trial_dict={'dataset_name':"FLChainDataset",'dataset_params':dataset_dict,'sequence_length':10,'prediction_length':10,'input_channels':1,'output_channels':1,'n_samples':5499,'n_timesteps':20, 'B':9,'epochs':200,'n_trials':10}

execute_trials(**trial_dict)

#StressStrainDataset
dataset_dict={  "lot": "all",
                "include_lot_as_feature": True,
                "downsample": True,
                "more_samples": 0,
                "specimen": "all",
                "max_strain": 0.2}

trial_dict={'dataset_name':"StressStrainDataset",'dataset_params':dataset_dict,'sequence_length':140,'prediction_length':70,'input_channels':1,'output_channels':1,'n_samples':47,'n_timesteps':210,
                "fixed_len":210, 'B':9,'epochs':200,'n_trials':10,'Ti':0.0,'T':1.0}

execute_trials(**trial_dict)

#Sines sum
print("Sines sum")
sequence_length = 40
prediction_length = 20
input_channels = 1
output_channels = 1
n_samples=200
n_timesteps=sequence_length+prediction_length

#generate a Sine dataset
Dataset_generator=SineTransDataset_AR(n_samples=n_samples, n_timesteps=n_timesteps)
X,ts,ys=Dataset_generator.get_X_ts_ys()

Ys1=np.array(ys)
X1_torch=torch.tensor(Ys1[:,:sequence_length]).float().reshape(n_samples,sequence_length,1)
Y1_torch=torch.tensor(Ys1[:,sequence_length:]).float().reshape(n_samples,prediction_length,1)


#generate a Sine dataset
Dataset_generator=SineDataset_AR(n_samples=n_samples, n_timesteps=n_timesteps)
X,ts,ys=Dataset_generator.get_X_ts_ys()
Ys2=np.array(ys)
X2_torch=torch.tensor(Ys2[:,:sequence_length]).float().reshape(n_samples,sequence_length,1)
Y2_torch=torch.tensor(Ys2[:,sequence_length:]).float().reshape(n_samples,prediction_length,1)


X_torch=torch.square(X1_torch+X2_torch)
Y_torch=torch.square(Y1_torch+Y2_torch)
print("TSMixer")
model= TSMixer(
    sequence_length=sequence_length,
    prediction_length=prediction_length,
    input_channels=input_channels,
)

mean_loss,std_loss=run_trials(
    train_tsmixer,
    model,
    X_torch,
    Y_torch,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=200,
    batch_size=32,
    lr=1e-2,
    device="cpu",
    n_trials=10,
    seed=42,
)

B=5
t=ts[0][sequence_length:]
# Flatten all time arrays together

bspline=BSplineBasis(n_basis=B,t_range=(t[0],t[-1]))
Phis = list(bspline.get_all_matrices(np.array(ts)[:,sequence_length:])) # (D, sequence_length, B)

for i in range(B):
    plt.plot(t,Phis[0][:,i])
plt.show()

latent_model = TSMixer(
    sequence_length=sequence_length,   # same as time steps in X
    prediction_length=B, # number of spline coefficients to predict
    input_channels=1,
)

print("TSMixer as Encoder")
mean_loss,std_loss=run_trials(
    train_latent_tsmixer,
    latent_model,
    X_torch,
    Y_torch,
    Phis=Phis,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=200,
    batch_size=32,
    lr=1e-2,
    device="cpu",
    n_trials=10,
    seed=42,
)
