export default function handler(req, res) {
  res.send(`
# ===============================
# 📘 Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===============================
# 🧩 Load and Preprocess Data
# ===============================
data = pd.read_csv("/content/Iris.csv")

# Drop unnecessary columns
data = data.drop(['Id', 'Species'], axis=1)

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert scaled data to DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])

# ===============================
# ⚙️ Apply K-Means Clustering
# ===============================
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(data_scaled_df)

# Add cluster labels to the DataFrame
data_scaled_df['Cluster'] = kmeans.labels_

# ===============================
# 📊 Display Cluster Centers
# ===============================
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
print("🔹 Cluster Centers:\n")
print(centroids)

# ===============================
# 🎨 Visualizations
# ===============================

# Pairplot to see clustering pattern
sns.pairplot(data_scaled_df, hue='Cluster', diag_kind='kde', palette='Set2')
plt.suptitle('K-Means Clusters on Iris Dataset', fontsize=14, y=1.02)
plt.show()

# 2D Scatter plot of first two features
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='SepalLength', y='SepalWidth',
    hue='Cluster', palette='Set2',
    data=data_scaled_df, s=70
)
plt.scatter(centroids['SepalLength'], centroids['SepalWidth'], 
            s=200, c='black', marker='X', label='Centroids')
plt.title('K-Means Clustering (2D View)')
plt.legend()
plt.show()

# ===============================
# 🧾 Optional: Evaluate with Inertia
# ===============================
print(f"Total Within-Cluster Sum of Squares (Inertia): {kmeans.inertia_:.3f}")

`);
}
