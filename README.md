
# Time Series Analysis of American Electric Power (AEP) Energy Consumption

## Project Overview
This project performs a time series analysis of hourly energy consumption data from American Electric Power (AEP). The analysis aims to understand energy usage patterns and potentially forecast future consumption.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Libraries Used](#libraries-used)
4. [Data Import and Preprocessing](#data-import-and-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Time Series Decomposition](#time-series-decomposition)
7. [Modeling and Forecasting](#modeling-and-forecasting)
8. [Results and Observations](#results-and-observations)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Dataset Information
- **Source**: [AEP hourly consumption dataset](https://raw.githubusercontent.com/khsieh18/Time-Series/master/AEP_hourly.csv)
- **Description**: The dataset contains hourly records of electricity consumption from AEP, including timestamps and measurements in megawatts (MW).
- **Columns**:
  - `Datetime`: Timestamp of each hourly observation.
  - `AEP_MW`: Electricity consumption in megawatts.

## Libraries Used
The following libraries are utilized for data manipulation, visualization, and time series analysis:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

## Data Import and Preprocessing
The dataset is imported directly from an online CSV file and loaded into a pandas DataFrame. Basic preprocessing steps include:
1. Parsing the `Datetime` column as a datetime object.
2. Setting `Datetime` as the index for time series operations.
3. Handling any missing values, if present, through imputation or removal.

Example:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/khsieh18/Time-Series/master/AEP_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df.head()
```

## Exploratory Data Analysis (EDA)
EDA aims to gain insights into the data's temporal structure and statistical properties:
- **Initial Inspection**: Checking the data's shape, data types, and summary statistics.
- **Visualization**: Plotting the time series to observe any visible trends, seasonality, or anomalies.
- **Statistics**: Calculating basic statistics (mean, median, etc.) for the `AEP_MW` values.

Example visualization:
```python
plt.figure(figsize=(14, 7))
plt.plot(df['AEP_MW'], label='AEP Energy Consumption')
plt.title('AEP Hourly Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MW)')
plt.legend()
plt.show()
```

## Time Series Decomposition
Using decomposition techniques to separate the series into trend, seasonality, and residual components, typically via:
- **Additive Decomposition**: For data where seasonality and trend components add up.
- **Multiplicative Decomposition**: For data where components multiply.

Example:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['AEP_MW'], model='additive', period=24)
decomposition.plot()
plt.show()
```

## Modeling and Forecasting
Several models can be applied for time series forecasting:
1. **Moving Average**: A simple smoothing technique for short-term trends.
2. **ARIMA**: Autoregressive Integrated Moving Average, a popular statistical model.
3. **Prophet**: For data with strong seasonal trends, Facebook's Prophet model can be used.
4. **LSTM/GRU Neural Networks**: For more complex and non-linear trends in data.

### Example: ARIMA Model
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model
model = ARIMA(df['AEP_MW'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

## Results and Observations
This section summarizes the findings from each model:
- **Trend Analysis**: Description of long-term consumption patterns.
- **Seasonal Patterns**: Insights into daily or weekly cycles in energy consumption.
- **Forecasting Performance**: Evaluation metrics for each model (e.g., RMSE, MAE).

## Conclusion
Summarize the project’s insights and suggest possible areas for further research or analysis, such as integrating weather data or exploring alternative models.

## References
- **Data Source**: [AEP hourly consumption dataset on GitHub](https://raw.githubusercontent.com/khsieh18/Time-Series/master/AEP_hourly.csv)
- **Time Series Analysis Techniques**: Reference any time series analysis resources or textbooks used.

---

This documentation provides a clear structure for understanding and replicating the analysis.
