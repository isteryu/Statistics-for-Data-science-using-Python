import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Create a sample dataset
data = {
'Age': [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
'Bought_Insurance': [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
# Step 2: Plot data
plt.scatter(df['Age'], df['Bought_Insurance'], color='red', marker='x')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.title("Age vs Bought Insurance")
plt.grid(True)
plt.show()
# Step 3: Train/Test Split
X = df[['Age']] # Feature must be in 2D
y = df['Bought_Insurance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize sigmoid curve
ages = np.linspace(20, 70, 200).reshape(-1, 1)
probabilities = model.predict_proba(ages)[:, 1]

plt.scatter(df['Age'], df['Bought_Insurance'], color='red', label="Actual")
plt.plot(ages, probabilities, color='blue', label="Logistic Curve")
plt.xlabel("Age")
plt.ylabel("Probability of Buying Insurance")
plt.title("Logistic Regression - Age vs Insurance Purchase")
plt.legend()
plt.grid(True)
plt.show()