export default function handler(req, res) {
  res.send(`

    # ===============================
# 🌳 Decision Tree with Plots
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load dataset ---
df = pd.read_csv("Iris.csv")

# Split data
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Feature scaling (optional for Decision Tree)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train model ---
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Accuracy & Reports ---
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 🌳 1️⃣ Visualize the Decision Tree
# ===============================
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=df.columns[1:-1],
    class_names=df["Species"].unique(),
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("🌳 Decision Tree Visualization", fontsize=16)
plt.show()

# ===============================
# 📊 2️⃣ Confusion Matrix Heatmap
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("📊 Confusion Matrix Heatmap", fontsize=14)
plt.show()

# ===============================
# 💡 3️⃣ Feature Importance Plot
# ===============================
importances = model.feature_importances_
features = df.columns[1:-1]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("💡 Feature Importance", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()



################################## updated #####################################

# =====================================================
# 🌳 Decision Tree Implementation and Parameter Analysis
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("Iris.csv")

# Drop Id column if exists
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data (optional for decision trees)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 2️⃣ ID3 Algorithm (criterion='entropy')
# -------------------------------
id3_model = DecisionTreeClassifier(
    criterion="entropy", random_state=42
)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

acc_id3 = accuracy_score(y_test, y_pred_id3)
print(f"✅ ID3 Accuracy: {acc_id3:.3f}")

# -------------------------------
# 3️⃣ C4.5 Algorithm (criterion='log_loss')
# -------------------------------
# sklearn uses 'log_loss' (similar to C4.5 gain ratio)
c45_model = DecisionTreeClassifier(
    criterion="log_loss", random_state=42
)
c45_model.fit(X_train, y_train)
y_pred_c45 = c45_model.predict(X_test)

acc_c45 = accuracy_score(y_test, y_pred_c45)
print(f"✅ C4.5 Accuracy: {acc_c45:.3f}")

# -------------------------------
# 4️⃣ Compare Parameter Effects
# -------------------------------
depths = [2, 3, 4, 5, 6, None]
min_samples_splits = [2, 5, 10]
min_samples_leafs = [1, 2, 4]

results = []

for depth in depths:
    for split in min_samples_splits:
        for leaf in min_samples_leafs:
            model = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            results.append({
                "max_depth": depth if depth else "None",
                "min_samples_split": split,
                "min_samples_leaf": leaf,
                "accuracy": acc
            })

result_df = pd.DataFrame(results)
print("\n📊 Parameter Comparison Table:")
print(result_df.head(10))

# -------------------------------
# 5️⃣ Visualization of Decision Tree
# -------------------------------
plt.figure(figsize=(14, 8))
plot_tree(
    id3_model,
    feature_names=df.columns[:-1],
    class_names=df["Species"].unique(),
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("🌳 ID3 Decision Tree Visualization", fontsize=14)
plt.show()

# -------------------------------
# 6️⃣ Confusion Matrix Heatmaps
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_id3), annot=True, cmap="Blues", fmt="d", ax=axes[0])
axes[0].set_title("ID3 Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_c45), annot=True, cmap="Greens", fmt="d", ax=axes[1])
axes[1].set_title("C4.5 Confusion Matrix")
plt.show()

# -------------------------------
# 7️⃣ Feature Importance Plot
# -------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    x=id3_model.feature_importances_,
    y=df.columns[:-1],
    palette="viridis"
)
plt.title("💡 Feature Importance (ID3)", fontsize=14)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# -------------------------------
# 8️⃣ Classification Reports
# -------------------------------
print("\n📘 ID3 Classification Report:\n", classification_report(y_test, y_pred_id3))
print("\n📗 C4.5 Classification Report:\n", classification_report(y_test, y_pred_c45))

# -------------------------------
# 9️⃣ Accuracy Comparison Bar Chart
# -------------------------------
plt.figure(figsize=(6, 5))
plt.bar(["ID3 (Entropy)", "C4.5 (Log Loss)"], [acc_id3, acc_c45], color=["blue", "green"])
plt.ylabel("Accuracy")
plt.title("📈 Accuracy Comparison: ID3 vs C4.5", fontsize=14)
plt.ylim(0.9, 1.0)
for i, v in enumerate([acc_id3, acc_c45]):
    plt.text(i, v - 0.02, f"{v:.3f}", color="white", fontweight="bold", ha="center")
plt.show()


# Parameter	Description	Effect on Accuracy
# criterion	Metric to measure the quality of a split.
# Options: 'gini' (CART default) or 'entropy' (similar to ID3).	Choosing 'entropy' gives splits like ID3/C4.5, 'gini' is faster and often similar in performance.
# max_depth	Maximum depth of the tree.	Prevents overfitting; smaller depth → simpler model, less overfit. Too small → underfit.
# min_samples_split	Minimum number of samples required to split an internal node.	Larger value → fewer splits → simpler model, helps control overfitting.
# min_samples_leaf	Minimum samples required at a leaf node.	Ensures leaf nodes have enough data, smoothing predictions.
# max_features	Number of features to consider at each split.	Reduces variance; used in ensemble models like Random Forest.
`);
}
