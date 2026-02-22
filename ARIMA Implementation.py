# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load dataset (Local file)
df = pd.read_excel(r"c:\Users\Dell\OneDrive\Desktop\- Statistics for Data science using Python\chennai_temperature (1).xlsx")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot original data
plt.figure(figsize=(10,4))
plt.plot(df['Temperature'], label='Temperature')
plt.title('Chennai Temperature over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# ACF and PACF before differencing
plot_acf(df['Temperature'], lags=min(40, len(df)//2 - 1))
plot_pacf(df['Temperature'], lags=min(40, len(df)//2 - 1))
plt.show()

# Differencing
df_diff = df['Temperature'].diff().dropna()

plt.figure(figsize=(10,4))
plt.plot(df_diff)
plt.title('Differenced Temperature Data')
plt.show()

# ACF and PACF after differencing
plot_acf(df_diff, lags=min(40, len(df_diff)//2 - 1))
plot_pacf(df_diff, lags=min(40, len(df_diff)//2 - 1))
plt.show()

# Build ARIMA model
model = ARIMA(df['Temperature'], order=(2,1,2))
model_fit = model.fit()

# Print summary
print(model_fit.summary())

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)

print("Forecasted Temperature:\n", forecast)

# Plot forecast
plt.figure(figsize=(10,4))
plt.plot(df['Temperature'], label='Historical')
plt.plot(forecast.index, forecast, label='Forecast')
plt.title('Temperature Forecast for Chennai')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()