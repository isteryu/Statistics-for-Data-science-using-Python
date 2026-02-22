# Simple Linear Regression using scikit-learn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("http://bit.ly/w-data") # Student Scores dataset
X = data[['Hours']]
y = data['Scores']

# Model creation
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)
# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.show()
# Evaluation
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))