## Overview
Traffic congestion is a widespread issue in urban areas worldwide, leading to increased travel times, fuel consumption, and environmental pollution. Economically, it causes inefficiencies and missed opportunities, while health-wise, it induces stress, fatigue, and respiratory issues due to prolonged exposure to emissions. Environmentally, it exacerbates air and noise pollution, contributing to climate change and reducing quality of life. Socially, congestion limits time with family and friends, diminishes community cohesion, and poses safety risks for pedestrians and cyclists. Urban infrastructure also suffers, requiring continuous investment and often driving urban sprawl. Addressing these challenges necessitates effective traffic management, investment in public transportation, and sustainable urban planning.

## Project Description
This project focuses on predicting traffic volume using advanced machine learning models. We use both traditional statistical methods, like the Seasonal Autoregressive Integrated Moving Average (SARIMA), and modern deep learning techniques, such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU). By analyzing historical traffic data, aim is to develop accurate prediction models that help in efficient traffic management and congestion reduction. Our approach includes creating hybrid models that combine the strengths of both statistical and deep learning methods, resulting in improved prediction accuracy and reliability. The ultimate goal is to provide actionable insights for better urban planning and traffic control, leading to reduced congestion, lower emissions, and a more efficient transportation system.

## What is done in this Project
#### Data Collection and Preparation 
- Gather historical traffic volume data from reliable sources such as traffic sensors and databases. Preprocess the data by handling missing values, normalizing or standardizing features, and splitting the data into training, validation, and test sets.

#### Exploratory Data Analysis (EDA)
- Analyze the dataset to understand its structure and identify patterns, trends, and seasonal behaviors in traffic volume. Visualize the data using graphs and charts to gain insights into temporal dynamics and correlations.
  
#### Model Implementation
- Implement traditional statistical models such as SARIMA to capture seasonal patterns in traffic data.
- Develop deep learning models including LSTM and GRU to handle complex temporal dependencies and nonlinear relationships. Build hybrid models that combine the outputs of SARIMA and LSTM/GRU to leverage the strengths of both approaches.

#### Training the Models
- Train the SARIMA model on the dataset by optimizing its parameters to accurately reflect seasonality and trends.
- Train the LSTM and GRU models using the training dataset, incorporating techniques like dropout and early stopping to prevent overfitting.
- For hybrid models, train the individual SARIMA and LSTM/GRU models separately before combining their predictions.
- Create ensemble models by averaging the predictions of SARIMA and LSTM/GRU models based on their respective performance weights. Apply regularization techniques during training to ensure that the ensemble model generalizes well to new data.
- Fine-tune the hyperparameters of the models to enhance their prediction accuracy.

  #### Rolling Forecasting
- Implement rolling forecasting to assess the real-world performance of the models by continuously updating predictions with new incoming data.
- Test the models in a real-time setting to ensure their robustness and adaptability to changing traffic conditions.
  
#### Model Evaluation
- Evaluate the performance of each model using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
- Compare the accuracy and reliability of the predictions from individual models (SARIMA, LSTM, GRU) and the hybrid models.

## TeckStack and Libraries Requirements
- Hardware: Google Compute Backend Engine TPU, RAM of 4.66GB and Disk size of 15.93GB
- Tools: VS code 
- Programming Language: Python
- Libraries: NumPy, Pandas, Seaborn, Matplotlib, SciKit-learn, Statsmodels (SARIMAX), pmdarima (auto_arima), tensorflow, keras, models(Sequential), layers(LSTM, Dense, Dropout, GRU)

## Dataset Description and EDA
I utilised [Metro InterState Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) dataset for this project. The dataset contains 48,204 entries. It has 9 columns 'Holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description', 'date_time', 'traffic_volume'. The dataset contains 4 object-type columns ('holiday', 'weather_main', 'weather_description', 'date_time'), 3 float64-type columns ('temp', 'rain_1h', 'snow_1h'), 2 int64-type columns ('clouds_all', 'traffic_volume'). Also 'holiday' column exists Missing values with only 61 non-null entries. 'date_time' column Provides temporal information for each observation.

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
The ensemble model combines predictions from SARIMA and stacked LSTM models, averaging their forecasts based on their respective performance weights.SARIMA and stacked LSTM models are individually trained on the same dataset to capture distinct data patterns effectively. Regularization techniques are applied during training to prevent overfitting, particularly for the more complex stacked LSTM model. Performance metrics such as RMSE, MAE, and MAPE quantify the accuracy and reliability of ensemble predictions against actual traffic volumes. The combination of SARIMA and stacked LSTM models enhances prediction
accuracy and stability in traffic forecasting.

### 6.) Hybrid Model of Stacked LSTM and SARIMA
The hybrid model combines forecasts from SARIMA and Stacked LSTM. SARIMA and Stacked LSTM are trained separately on the same dataset, with SARIMA parameters optimized for seasonality and Stacked LSTM fine-tuned to capture complex temporal dependencies. It Utilizes rolling forecasting to assess real-world performance. Dropout and early stopping employed in Stacked LSTM to prevent overfitting. It Outperforms individual models in forecasting accuracy across different traffic situations.vThe Hybrid SARIMA and Stacked LSTM model improves forecasting performance by leveraging unique qualities of both statistical and deep learning models. It enhances accuracy and produces solid predictions under varying traffic conditions.

## Results
- MAE (mean absolute error) and RMSE (root mean squared error) of LSTM were 341.51 and 560.46
- MAE (mean absolute error) and RMSE (root mean squared error) of Stacked LSTM were 338.62 and 535.62
- MAE (mean absolute error) and RMSE (root mean squared error) of GRU were 562.22 and 779.12
- MAE (mean absolute error) and RMSE (root mean squared error) of SARIMA were 308.80 and 453.88
- MAE (mean absolute error) and RMSE (root mean squared error) of SARIMA and stacked LSTM hybrid were 273.26 and 398.48
- MAE (mean absolute error) and RMSE (root mean squared error) of Ensemble learning of SARIMA and stacked LSTM were 1198.14 and 1474.71

## Mitigation Techniques
Prediction and Early Intervention: These models, especially the hybrid SARIMA and LSTM model, have proven effective in predicting traffic flow and congestion events with high accuracy. This predictive capability is crucial for early intervention. By accurately forecasting traffic volume and identifying potential congestion before it happens, traffic management systems can implement measures to prevent congestion from occurring in the first place.
- Dynamic Traffic Management: With real-time data and predictive insights from your models, traffic management systems can dynamically adjust signal timings and traffic flow directions to alleviate congestion points before they become problematic. This proactive approach shifts from reactive to preventive traffic management.
- Optimization of Traffic Signals and Routes: traffic signals can be optimized to ensure smoother flow of traffic. For example, extending green light durations at busy intersections predicted to experience high traffic volumes can reduce the likelihood of bottlenecks.
