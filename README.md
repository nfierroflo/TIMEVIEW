# TIMEVIEW

This is  a modfied version of the original repository for the paper "Towards Transparent Time Series Forecasting".
This versions implements the TIMEVIEW4FE model.

## Clone the repository
Clone the repository using
```
git clone https://github.com/nfierroflo/TIMEVIEW.git
```

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n timeview --file environment.yml
```
This will also install `timeview` (the main module) in editable mode.

## Running all experiments
To run TIMEVIEW4FE experiments navigate to `TIMEVIEW4FE` using
```
cd TIMEVIEW4FE
``` 
and run
```
./run_scripts/run_all.sh
```
Or you can call the scripts individually in `run_scripts`.

The results will be saved in
```
TIMEVIEW4FE/results
```

## Citations
If you use this code, please cite using the following information.

*Kacprzyk, K., Liu, T. & van der Schaar, M. Towards Transparent Time Series Forecasting. The Twelfth International Conference on Learning Representations (2024).*


```
@inproceedings{Kacprzyk.TransparentTimeSeries.2024,
  title = {Towards Transparent Time Series Forecasting},
  booktitle = {The {{Twelfth International Conference}} on {{Learning Representations}}},
  author = {Kacprzyk, Krzysztof and Liu, Tennison and {van der Schaar}, Mihaela},
  year = {2024},
}
```

