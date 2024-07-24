## Overview

## Project Description

## What is done in this Project

## TeckStack and Libraries Requirements
- Hardware:
- Tools: VS code 
- Programming Language: Python
- Libraries: NumPy, Pandas, Seaborn, Matplotlib

## Dataset Description and EDA
I utilised [Metro InterState Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) dataset for this project. 

## Models Utilised

### 1.) LSTM
- LSTMs are a special kind of RNN — capable of learning long-term dependencies by remembering information for long periods is the default behavior.
- All RNN are in the form of a chain of repeating modules of a neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.
- LSTMs also have a chain-like structure, but the repeating module is a bit different structure. Instead of having a single neural network layer, four interacting layers are communicating extraordinarily.
![image](https://github.com/user-attachments/assets/5dcd663c-dd5d-4b32-a7f8-2ce1c4e22e25)

### 2.) GRU
Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture that is used for modeling sequential data. It is designed to handle the vanishing gradient problem, which is common in traditional RNNs. GRUs are similar to Long Short-Term Memory (LSTM) networks but are simpler and have fewer parameters. Key Components of GRU:
- Update Gate (z): Controls how much of the previous state needs to be passed along to the future. It decides what information should be carried forward to the next time step. Controls the balance between the previous hidden state and the new candidate hidden state.
- Reset Gate (r): Decides how much of the past information to forget. It determines what part of the previous state to discard.Controls how much past information is considered for the current candidate hidden state.
- Current Memory Content: Uses the reset gate to store relevant information from the past to make predictions.
- Candidate Hidden State (ℎ~𝑡): The new content to be added to the network's memory.
- Hidden State (h_t): The output state of the GRU at time t, a combination of the previous hidden state and the candidate hidden state based on the update gate.
![image](https://github.com/user-attachments/assets/51af1357-0231-4fdc-9b56-96af22ba0c04)

### 3.) Stacked LSTM


### 4.) SARIMA
SARIMA, which stands for Seasonal Autoregressive Integrated Moving Average, is a versatile and widely used time series forecasting model. It’s an extension of the non-seasonal ARIMA model, designed to handle data with seasonal patterns. SARIMA captures both short-term and long-term dependencies within the data, making it a robust tool for forecasting. It combines the concepts of autoregressive (AR), integrated (I), and moving average (MA) models with seasonal components. The Components of SARIMA: 
- Seasonal Component: The “S” in SARIMA represents seasonality, which refers to repeating patterns in the data. This could be daily, monthly, yearly, or any other regular interval. Identifying and modelling the seasonal component is a key strength of SARIMA.
- Autoregressive (AR) Component: The “AR” in SARIMA signifies the autoregressive component, which models the relationship between the current data point and its past values. It captures the data’s autocorrelation, meaning how correlated the data is with itself over time.
- Integrated (I) Component: The “I” in SARIMA indicates differencing, which transforms non-stationary data into stationary data. Stationarity is crucial for time series modelling. The integrated component measures how many differences are required to achieve stationarity.
- Moving Average (MA) Component: The “MA” in SARIMA represents the moving average component, which models the dependency between the current data point and past prediction errors. It helps capture short-term noise in the data.

![image](https://github.com/user-attachments/assets/143b3df0-f887-48f6-b49e-d88b92d3b9f3)

### 5.) Ensemble Learning of LSTM and SARIMA

### 6.) Hybrid Model of Stacked LSTM and SARIMA

## Results

## Mitigation Techniques
