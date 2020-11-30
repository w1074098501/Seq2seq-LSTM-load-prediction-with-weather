# Sequence to sequence LSTM load forecasting with weather features
This project was made for final project in Artificial Intelligence and Machine Learning Applications in Modern Power System of Pitt

The project consists of two main contributions:
- Autoencoder for meteorological feature extraction
- Seq2seq LSTM models for load prediction with or without consideration of weather conditions

## Data
The dataset was collected from National Ocenic and Atomospheric Administration at https://www.ncdc.noaa.gov/cdo-web/datatools/findstation

Load data in Pittsburgh was collected from PJM hourly load data at https://dataminer2.pjm.com/feed/hrl_load_metered

## Training
Lauch autoencoder training by:
```python
python main_Autoencoder.py
```

Launch Load prediction training without weather conditions:
```python
python main_RNN_lag.py
```

Launch load prediction training without weather condictions:
```python
python main_RNN_lag_withoutweat.py
```