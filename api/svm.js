export default function handler(req, res) {
  res.send(`

    # ======================================================
# 🌸 SVM Classification (SVC) and Regression (SVR)
# on the Iris Dataset — with 3D Visualization
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVR
from sklearn.model_selection import GridSearchCV

from mpl_toolkits.mplot3d import Axes3D

# ======================================================
# 1️⃣ Load Dataset
# ======================================================
data = load_iris()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Feature names:", data.feature_names)
print("Target classes:", data.target_names)

# ======================================================
# 2️⃣ SVM CLASSIFICATION (SVC)
# ======================================================
print("\n=========== 🌳 SVM CLASSIFICATION (SVC) ===========")

# Use first 3 features for visualization
Xc = X[:, :3]
yc = y

# Define hyperparameter grid for C
param_grid_svc = {'C': [0.01, 0.1, 1, 10]}

# Create model and perform GridSearchCV (CV=5)
svc = SVC(kernel='linear')
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5)
grid_svc.fit(Xc, yc)

# Best model parameters
print("Best Parameters (SVC):", grid_svc.best_params_)

# Extract best model coefficients
w = grid_svc.best_estimator_.coef_[0]
b = grid_svc.best_estimator_.intercept_[0]

# Define 3D decision plane
x_min, x_max = Xc[:, 0].min(), Xc[:, 0].max()
y_min, y_max = Xc[:, 1].min(), Xc[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                     np.linspace(y_min, y_max, 30))
zz = -(w[0]*xx + w[1]*yy + b) / w[2]

# Plot SVM Decision Plane
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
for i, color, label in zip(range(3), colors, data.target_names):
    ax.scatter(Xc[yc == i, 0], Xc[yc == i, 1], Xc[yc == i, 2],
               c=color, label=label, s=50)

ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title("🌳 Linear SVC Decision Plane")
ax.legend()
plt.show()


# ======================================================
# 3️⃣ SVM REGRESSION (LinearSVR)
# ======================================================
print("\n=========== 📈 SVM REGRESSION (LinearSVR) ===========")

# Use different subset of features for regression
# (Predict feature_0 using features 1,2,3)
Xr = X[:, 1:4]
yr = X[:, 0]

# Define hyperparameter grid
param_grid_svr = {
    'C': [0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5]
}

# Create and train SVR model
svr = LinearSVR(max_iter=5000, random_state=1)
grid_svr = GridSearchCV(svr, param_grid_svr, cv=5)
grid_svr.fit(Xr, yr)

# Best parameters
print("Best Parameters (LinearSVR):", grid_svr.best_params_)

# Extract coefficients and intercept
w = grid_svr.best_estimator_.coef_
b = grid_svr.best_estimator_.intercept_[0]

# Define regression plane
x_min, x_max = Xr[:, 0].min(), Xr[:, 0].max()
y_min, y_max = Xr[:, 1].min(), Xr[:, 1].max()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                     np.linspace(y_min, y_max, 30))

# Approximate regression surface
zz = -(w[0]*xx + w[1]*yy - yr.mean() + b) / w[2]

# Plot Regression Plane
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xr[:, 0], Xr[:, 1], yr, c='g', s=20, label='Data Points')
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title("📈 Linear SVR Regression Plane")
ax.legend()
plt.show()



`);
}
