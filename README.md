# RainFAll_Prediction-Model
Machine learning-based rainfall prediction system using historical weather data. Implements models to analyze patterns in temperature, humidity, and pressure to forecast rainfall. Evaluates performance using accuracy metrics and helps in improving agricultural planning and weather decision-making.
#1st type using random forest regressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv("/content/rainfall_dataset.csv")
data.isnull().sum()
data.fillna(data.mean(), inplace=True)
X_features = data.drop("rainfall_mm", axis=1)
y = data["rainfall_mm"]

scaler=StandardScaler()
Scaled_X_features=scaler.fit_transform(X_features)

x=Scaled_X_features
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R-squared:",r2)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2)
new_data = [[
    30,   # temperature
    80,   # humidity
    1012, # pressure
    10,   # wind speed
    180,  # wind direction
    60,   # cloud cover
    25,   # dew point
    500,  # solar radiation
    0.3,  # soil moisture
    5,    # previous rainfall
    7,    # month
    23.5, # latitude
    88.3, # longitude
    10    # elevation
]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Predicted Rainfall (mm):", prediction[0])
