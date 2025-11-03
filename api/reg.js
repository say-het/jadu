export default function handler(req, res) {
  res.send(`
# ============================================================
# 🔹 Imports
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# --- Simple Regression Dataset ---
data_simple = pd.read_csv("Salary_Data.csv")   # Columns: YearsExperience, Salary
X_simple = data_simple["YearsExperience"].values
y_simple = data_simple["Salary"].values

# --- Multiple Regression Dataset ---
data_multi = pd.read_csv("Advertising.csv")    # Columns: TV, Radio, Newspaper, Sales
X_multi = data_multi[["TV", "Radio", "Newspaper"]].values
y_multi = data_multi["Sales"].values

x = data_multi.drop(columns=['medv']).values
y = data_multi['medv'].values

# ============================================================
# 🧮 PART 1: SIMPLE LINEAR REGRESSION (Without sklearn)
# ============================================================

print("\n=== SIMPLE LINEAR REGRESSION (NO SKLEARN) ===")

X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)
m = len(y_train)
w, b = 0, 0
alpha = 0.0001
epochs = 5000
losses = []

for _ in range(epochs):
    y_pred = w * X_train + b
    dw = (-2/m) * np.sum(X_train * (y_train - y_pred))
    db = (-2/m) * np.sum(y_train - y_pred)
    w -= alpha * dw
    b -= alpha * db
    losses.append(np.mean((y_train - y_pred)**2))

# Evaluate
y_pred_test = w * X_test + b
print("Weight:", w, "Bias:", b)
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, w*X_train + b, color='red')
plt.title("Simple Linear Regression Line")

plt.subplot(1,2,2)
plt.plot(losses)
plt.title("Loss Curve (MSE vs Epoch)")
plt.show()


# ============================================================
# 🧮 PART 2: MULTIPLE LINEAR REGRESSION (Without sklearn)
# ============================================================

print("\n=== MULTIPLE LINEAR REGRESSION (NO SKLEARN) ===")

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
m, n = X_train.shape
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
    losses.append(np.mean((y_train - y_pred)**2))

y_pred_test = X_test @ W + b
print("Weights:", W)
print("Bias:", b)
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

plt.plot(losses)
plt.title("Multiple Linear Regression Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()


# ============================================================
# 🧩 PART 3: LINEAR REGRESSION WITH REGULARIZATION (No sklearn)
# ============================================================
# We'll use Ridge-like L2 regularization: 
# Loss = MSE + λ * ||W||²

print("\n=== REGULARIZED LINEAR REGRESSION (NO SKLEARN) ===")

lam = 0.1   # Regularization parameter
W = np.zeros(n)
b = 0
alpha = 0.00001
epochs = 5000
losses = []

for _ in range(epochs):
    y_pred = X_train @ W + b
    dW = (-2/m) * (X_train.T @ (y_train - y_pred)) + 2 * lam * W
    db = (-2/m) * np.sum(y_train - y_pred)
    W -= alpha * dW
    b -= alpha * db
    loss = np.mean((y_train - y_pred)**2) + lam * np.sum(W**2)
    losses.append(loss)

y_pred_test = X_test @ W + b
print("Weights:", W)
print("Bias:", b)
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

plt.plot(losses)
plt.title("Ridge (L2) Regularized Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# ============================================================
# ⚙️ PART 4: USING SKLEARN (Simple & Multiple)
# ============================================================

from sklearn.linear_model import LinearRegression, Ridge, Lasso

print("\n=== SIMPLE LINEAR REGRESSION (SKLEARN) ===")
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_simple.reshape(-1,1), y_simple, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_s_train, y_s_train)
y_pred = model.predict(X_s_test)
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("MSE:", mean_squared_error(y_s_test, y_pred))
print("R²:", r2_score(y_s_test, y_pred))

plt.scatter(y_s_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Sklearn Simple Linear Regression")
plt.show()

print("\n=== MULTIPLE LINEAR REGRESSION (SKLEARN) ===")
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_m_train, y_m_train)
y_pred = model.predict(X_m_test)
print("Intercept:", model.intercept_)
print("Weights:", model.coef_)
print("MSE:", mean_squared_error(y_m_test, y_pred))
print("R²:", r2_score(y_m_test, y_pred))

plt.scatter(y_m_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sklearn Multiple Linear Regression")
plt.show()


# ============================================================
# 🧠 PART 5: REGULARIZED LINEAR REGRESSION (SKLEARN)
# ============================================================

print("\n=== RIDGE (L2) REGULARIZATION (SKLEARN) ===")
ridge = Ridge(alpha=1.0)
ridge.fit(X_m_train, y_m_train)
y_pred_ridge = ridge.predict(X_m_test)
print("Intercept:", ridge.intercept_)
print("Weights:", ridge.coef_)
print("MSE:", mean_squared_error(y_m_test, y_pred_ridge))
print("R²:", r2_score(y_m_test, y_pred_ridge))

print("\n=== LASSO (L1) REGULARIZATION (SKLEARN) ===")
lasso = Lasso(alpha=0.1)
lasso.fit(X_m_train, y_m_train)
y_pred_lasso = lasso.predict(X_m_test)
print("Intercept:", lasso.intercept_)
print("Weights:", lasso.coef_)
print("MSE:", mean_squared_error(y_m_test, y_pred_lasso))
print("R²:", r2_score(y_m_test, y_pred_lasso))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_m_test, y_pred_ridge, color='green')
plt.title("Ridge Regression (Predicted vs Actual)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1,2,2)
plt.scatter(y_m_test, y_pred_lasso, color='orange')
plt.title("Lasso Regression (Predicted vs Actual)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

`);
}
