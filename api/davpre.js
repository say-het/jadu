export default function handler(req, res) {
  res.send(`

# 🧹 Data Cleaning Script
# Handles missing values, outliers, and duplicates automatically.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load your dataset
# -----------------------------
# Example: Replace with your file name
df = pd.read_csv("your_dataset.csv")

print("🔹 Original Data Shape:", df.shape)
print("\n🔹 Missing Values Before Cleaning:\n", df.isnull().sum())

# -----------------------------
# 2️⃣ Remove duplicate rows
# -----------------------------
df.drop_duplicates(inplace=True)
print("\n✅ Duplicates removed. New shape:", df.shape)

# -----------------------------
# 3️⃣ Handle Missing Values
# -----------------------------
# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Fill numeric missing values with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical missing values with mode
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values handled successfully!")
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# -----------------------------
# 4️⃣ Handle Outliers (IQR method)
# -----------------------------
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_limit) & (data[column] <= upper_limit)]

before = df.shape[0]
for col in num_cols:
    df = remove_outliers_iqr(df, col)
after = df.shape[0]

print(f"\n✅ Outliers handled using IQR. Rows reduced from {before} → {after}")

# -----------------------------
# 5️⃣ Check Final Summary
# -----------------------------
print("\n📊 Final Cleaned Data Info:")
print(df.info())
print("\n📈 Basic Stats:")
print(df.describe())

# -----------------------------
# 6️⃣ (Optional) Visualize
# -----------------------------
# Boxplots to see outliers visually
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# -----------------------------
# 7️⃣ Save Cleaned Dataset
# -----------------------------
df.to_csv("cleaned_dataset.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_dataset.csv'")

`);
}
