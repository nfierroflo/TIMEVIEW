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

#create a dataframe for results, with dataset, model, mean_loss, std_loss
results=pd.DataFrame(columns=['dataset','model','mean_loss','std_loss'])

def execute_trials(dataset_params,dataset_name,n_samples, n_timesteps, B, sequence_length, prediction_length, input_channels, output_channels, epochs, n_trials,fixed_len=None,T=None,Ti=None):
    global results
    #call the dataset generator
    Dataset_generator=eval(dataset_name)(**dataset_params)
    print(f"{dataset_name}")
    #generate a dataset
    if fixed_len is not None:
        X,ts,ys=Dataset_generator.get_X_ts_ys(fixed_len)
    else:
        X,ts,ys=Dataset_generator.get_X_ts_ys()

    Ys=np.array(ys)
    X_torch=torch.tensor(X.values).float().reshape(n_samples,sequence_length,input_channels)
    Y_torch=torch.tensor(Ys).float().reshape(n_samples,prediction_length,1)

    
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

    #add this result as a new row to the results dataframe
    new_row = pd.DataFrame({'dataset': [dataset_name], 'model': ['TSMixer'], 'mean_loss': [mean_loss], 'std_loss': [std_loss]})
    # Concatenate the new row
    results = pd.concat([results, new_row], ignore_index=True)


    internal_knots=calculate_knot_placement(ts, ys, n_internal_knots=B-2, T=1,seed=0, verbose=False)

    bspline=BSplineBasis(n_basis=B,t_range=(0.0,1.0),internal_knots=internal_knots)
    Phis = list(bspline.get_all_matrices(np.array(ts)))

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
        n_trials=n_trials,
        seed=42,
    )

    #add this result as a new row to the results dataframe
    new_row = pd.DataFrame({'dataset': [dataset_name], 'model': ['TIMEVIEW4FE(TSMixer)'], 'mean_loss': [mean_loss], 'std_loss': [std_loss]})
    # Concatenate the new row
    results = pd.concat([results, new_row], ignore_index=True)
n_trials=10

#SineTransDataset
dataset_dict={'n_samples':200,'n_timesteps':20}
trial_dict={'dataset_name':"SineTransDataset",'dataset_params':dataset_dict,'sequence_length':1,'prediction_length':20,'input_channels':1,'output_channels':1,'n_samples':200,'n_timesteps':20, 'B':5,'epochs':200,'n_trials':n_trials}

execute_trials(**trial_dict)

dataset_dict={'n_samples':900,'n_timesteps':20}
trial_dict={'dataset_name':"BetaDataset",'dataset_params':dataset_dict,'sequence_length':1,'prediction_length':20,'input_channels':2,'output_channels':1,'n_samples':900,'n_timesteps':20, 'B':5,'epochs':200,'n_trials':n_trials}

execute_trials(**trial_dict)

dataset_dict={
'n_samples': 2000,'n_time_steps': 20,'time_horizon': 1.0,'noise_std': 0.0,'seed': 0,'equation': 'wilkerson'
}
trial_dict={'dataset_name':"SyntheticTumorDataset",'dataset_params':dataset_dict,'sequence_length':1,'prediction_length':20,'input_channels':4,'output_channels':1,'n_samples':2000,'n_timesteps':20, 'B':9,'epochs':200,'n_trials':n_trials}

execute_trials(**trial_dict)

#FLChainDataset
os.chdir('../experiments')
Dataset_generator=FLChainDataset()
print("FLChainDataset")
X,ts,ys=Dataset_generator.get_X_ts_ys()
X['sex'] = X['sex'].replace({'F': 0, 'M': 1})
X['mgus'] = X['mgus'].replace({'no': 0, 'yes': 1})

sequence_length = 1
prediction_length = 20
input_channels = X.shape[-1]
output_channels = 1

Ys=np.array(ys)
Ts=np.array(ts)


X_torch=torch.tensor(X.values).float().reshape(-1,sequence_length,input_channels)
Y_torch=torch.tensor(Ys).float().reshape(-1,prediction_length,output_channels)


model= TSMixer(
    sequence_length=sequence_length,
    prediction_length=prediction_length,
    input_channels=input_channels,
    output_channels=output_channels,
)

print("TSMixer")
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
    n_trials=n_trials,
    seed=42,
)

#add this result as a new row to the results dataframe
new_row = pd.DataFrame({'dataset': ['FLChainDataset'], 'model': ['TSMixer'], 'mean_loss': [mean_loss], 'std_loss': [std_loss]})
# Concatenate the new row
results = pd.concat([results, new_row], ignore_index=True)

B=9
ts_N=[arr[sequence_length:] for arr in ts]
ys_N=[arr[sequence_length:] for arr in ys]

internal_knots=calculate_knot_placement(ts_N, ys_N, T=1.0, n_internal_knots=B-2, seed=0, verbose=False)

bspline=BSplineBasis(n_basis=B,t_range=(0.0,1.0),internal_knots=internal_knots)
Phis = list(bspline.get_all_matrices(np.array(Ts)))

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
    n_trials=n_trials,
    seed=42,
)
#add this result as a new row to the results dataframe
new_row = pd.DataFrame({'dataset': ['FLChainDataset'], 'model': ['TIMEVIEW4FE(TSMixer)'], 'mean_loss': [mean_loss], 'std_loss': [std_loss]})
# Concatenate the new row
results = pd.concat([results, new_row], ignore_index=True)

#StressStrainDataset
print("StressStrainDataset")
Dataset_generator=StressStrainDataset(**{
                "lot": "all",
                "include_lot_as_feature": True,
                "downsample": True,
                "more_samples": 0,
                "specimen": "all",
                "max_strain": 0.2})
X,ts,ys=Dataset_generator.get_X_ts_ys()

sequence_length = 1
prediction_length = 212

Ys=np.array([y[:prediction_length] for y in ys if len(y)>=prediction_length])
Ts=np.array([t[:prediction_length] for t in ts if len(t)>=prediction_length])
X=X[X.index.isin([i for i in range(len(ys)) if len(ys[i])>=prediction_length])]

transformer=Dataset_generator.get_default_column_transformer(keep_categorical=False)
X_transformed = transformer.fit_transform(X)

input_channels = X_transformed.shape[-1]
output_channels = 1


X_torch=torch.tensor(X_transformed).float().reshape(-1,sequence_length,input_channels)
Y_torch=torch.tensor(Ys).float().reshape(-1,prediction_length,output_channels)

model= TSMixer(
    sequence_length=sequence_length,
    prediction_length=prediction_length,
    input_channels=input_channels,
    output_channels=output_channels,
)

print("TSMixer")
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
    n_trials=n_trials,
    seed=42,
)

#add this result as a new row to the results dataframe
new_row = pd.DataFrame({'dataset': ['StressStrainDataset'], 'model': ['TSMixer'], 'mean_loss': [mean_loss], 'std_loss': [std_loss]})
# Concatenate the new row
results = pd.concat([results, new_row], ignore_index=True)

B=9
ts_N=[arr for arr in Ts]
ys_N=[arr for arr in Ys]

internal_knots=calculate_knot_placement(ts_N, ys_N, n_internal_knots=B-2, T=1 ,seed=0, verbose=False)

bspline=BSplineBasis(n_basis=B,t_range=(0,1),internal_knots=internal_knots)
Phis = list(bspline.get_all_matrices(np.array(Ts)))

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
    n_trials=n_trials,
    seed=42,
)

#add this result as a new row to the results dataframe
new_row = pd.DataFrame({'dataset': ['StressStrainDatase'], 'model': ['TIMEVIEW4FE(TSMixer)'], 'mean_loss': [mean_loss], 'std_loss': [std_loss]})
# Concatenate the new row
results = pd.concat([results, new_row], ignore_index=True)

os.chdir('../TIMEVIEW4FE')
#save the results to a csv file
results.to_csv('results/static_DS_results.csv',index=False)