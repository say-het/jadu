export default function handler(req, res) {
  res.send(`
# ===============================================
# 📘 Simple Linear Regression - Gradient Descent, Normal Equation & Sklearn
# ===============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# 1️⃣ Load dataset (e.g., Salary_Data.csv)
# ---------------------------------------------------
# Dataset should have columns: "YearsExperience", "Salary"
data = pd.read_csv("Salary_Data.csv")

# Split features and labels
X = data["YearsExperience"].values
y = data["Salary"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
m = len(y_train)

# ===============================================
# 🌈 1. Gradient Descent (No sklearn)
# ===============================================

w, b = 0, 0
alpha = 0.0001   # Learning rate
epochs = 10000
losses = []

for _ in range(epochs):
    y_pred = w * X_train + b
    dw = (-2/m) * np.sum(X_train * (y_train - y_pred))
    db = (-2/m) * np.sum(y_train - y_pred)
    w -= alpha * dw
    b -= alpha * db
    loss = np.mean((y_train - y_pred)**2)
    losses.append(loss)

print("Gradient Descent → Weight:", round(w,3), "Bias:", round(b,3))

# Predictions and metrics
y_pred_test = w * X_test + b
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R² Score:", r2_score(y_test, y_pred_test))

# 📊 Plots
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, w*X_train + b, color='red')
plt.title("Gradient Descent Regression Line")

plt.subplot(1,2,2)
plt.plot(losses)
plt.title("Loss Curve (MSE vs Epochs)")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

# ===============================================
# 🧮 2. Normal Equation (No sklearn)
# ===============================================
X_b = np.c_[np.ones((len(X_train), 1)), X_train]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
intercept, slope = theta[0], theta[1]
print("Normal Equation → Intercept:", round(intercept,3), "Slope:", round(slope,3))

# Evaluate
y_pred_ne = intercept + slope * X_test
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R² Score:", r2_score(y_test, y_pred_ne))

# Plot
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, slope*X_test + intercept, color='red')
plt.title("Normal Equation Regression Line")
plt.show()

# ===============================================
# 🤖 3. Using sklearn
# ===============================================
model = LinearRegression()
model.fit(X_train.reshape(-1,1), y_train)
y_pred_sk = model.predict(X_test.reshape(-1,1))

print("Sklearn → Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("MSE:", mean_squared_error(y_test, y_pred_sk))
print("R² Score:", r2_score(y_test, y_pred_sk))

# Plot comparison
plt.scatter(y_test, y_pred_sk, color='purple')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted (Sklearn)")
plt.show()




# ===============================================
# 📘 Multiple Linear Regression - Gradient Descent, Normal Equation & Sklearn
# ===============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# 1️⃣ Load dataset (e.g., Advertising.csv)
# ---------------------------------------------------
# Dataset should have columns: "TV", "Radio", "Newspaper", "Sales"
data = pd.read_csv("Advertising.csv")

X = data[["TV", "Radio", "Newspaper"]].values
y = data["Sales"].values

x = data.drop(columns=['medv']).values
y = data['medv'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
m, n = X_train.shape

# ===============================================
# 🌈 1. Gradient Descent
# ===============================================
W = np.zeros(n)
b = 0
alpha = 0.00001
epochs = 5000
losses = []

for _ in range(epochs):
    y_pred = X_train @ W + b
    dW = (-2/m) * (X_train.T @ (y_train - y_pred))
    db = (-2/m) * np.sum(y_train - y_pred)
    W -= alpha * dW
    b -= alpha * db
    loss = np.mean((y_train - y_pred)**2)
    losses.append(loss)

print("Gradient Descent → Weights:", W)
print("Bias:", b)

# Evaluate
y_pred_test = X_test @ W + b
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R² Score:", r2_score(y_test, y_pred_test))

# Plot loss curve
plt.plot(losses)
plt.title("Loss Curve (MSE vs Epochs)")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

# ===============================================
# 🧮 2. Normal Equation
# ===============================================
X_b = np.c_[np.ones((len(X_train), 1)), X_train]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train

intercept = theta[0]
weights = theta[1:]
print("Normal Equation → Intercept:", intercept)
print("Weights:", weights)

# Evaluate
y_pred_ne = X_test @ weights + intercept
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R² Score:", r2_score(y_test, y_pred_ne))

# ===============================================
# 🤖 3. Using sklearn
# ===============================================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sk = model.predict(X_test)

print("Sklearn → Intercept:", model.intercept_)
print("Weights:", model.coef_)
print("MSE:", mean_squared_error(y_test, y_pred_sk))
print("R² Score:", r2_score(y_test, y_pred_sk))

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred_sk, color='orange')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted (Sklearn)")
plt.show()



`);
}
