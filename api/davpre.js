export default function handler(req, res) {
  res.send(`

# =====================================
# Universal Data Analysis & Preprocessing Script
# =====================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

# ==========================
# STEP 1: LOAD DATASET
# ==========================
path = input("Enter dataset path (CSV/XLSX) or leave empty to use sample health dataset: ")

if path:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type! Please use CSV or XLSX.")
else:
    # Sample healthcare dataset if no path given
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame

print("\n✅ Dataset Loaded!")
print("Shape:", df.shape)
print(df.head())

# ==========================
# STEP 2: DATA INSPECTION
# ==========================
print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ==========================
# STEP 3: CLASSES & ATTRIBUTES
# ==========================
print("\n===== CLASSES & ATTRIBUTES =====")
for col in df.columns:
    unique_vals = df[col].nunique()
    if unique_vals <= 10:
        print(f"Possible class/label attribute: {col} -> {unique_vals} unique values")
    else:
        print(f"Possible continuous attribute: {col}")

# ==========================
# STEP 4: MISSING VALUES (UNCERTAINTY)
# ==========================
print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()
print(missing[missing > 0])

# --------- Handling Missing Values Variants ---------
# 1. Remove rows with missing values
df_dropna_rows = df.dropna()

# 2. Remove columns with missing values
df_dropna_cols = df.dropna(axis=1)

# 3. Fill with mean (numeric)
df_fill_mean = df.fillna(df.mean(numeric_only=True))

# 4. Fill with median (numeric)
df_fill_median = df.fillna(df.median(numeric_only=True))

# 5. Fill with mode (categorical or numeric)
df_fill_mode = df.fillna(df.mode().iloc[0])

# 6. Fill with a constant
df_fill_const = df.fillna(0)

print("\n✅ Missing values can be handled using various methods: drop rows/cols, fill mean/median/mode/constant")

# ==========================
# STEP 5: FIVE NUMBER SUMMARY
# ==========================
print("\n===== FIVE NUMBER SUMMARY =====")
five_num_summary = df.describe().loc[['min', '25%', '50%', '75%', 'max']]
print(five_num_summary)

# ==========================
# STEP 6: MODE & MIDRANGE
# ==========================
print("\n===== MODE & MIDRANGE =====")
mode_values = df.mode().iloc[0]
midrange = {}
for col in df.select_dtypes(include=[np.number]).columns:
    midrange[col] = (df[col].min() + df[col].max()) / 2

print("Mode:\n", mode_values)
print("\nMidrange:\n", pd.Series(midrange))

# ==========================
# STEP 7: OUTLIER DETECTION
# ==========================
print("\n===== OUTLIER DETECTION =====")

# --------- 1. IQR METHOD ---------
outliers_iqr = {}
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    outliers_iqr[col] = len(outliers)

print("\nIQR Outliers:\n", pd.Series(outliers_iqr))

# --------- 2. Z-Score METHOD ---------
outliers_zscore = {}
for col in df.select_dtypes(include=[np.number]).columns:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers = df[z_scores > 3][col]
    outliers_zscore[col] = len(outliers)

print("\nZ-Score Outliers:\n", pd.Series(outliers_zscore))

# --------- 3. Modified Z-Score (Median Absolute Deviation) ---------
outliers_mod_z = {}
for col in df.select_dtypes(include=[np.number]).columns:
    median_val = np.median(df[col])
    mad = np.median(np.abs(df[col] - median_val))
    if mad == 0:  # avoid division by zero
        outliers_mod_z[col] = 0
        continue
    modified_z_score = 0.6745 * (df[col] - median_val) / mad
    outliers = df[np.abs(modified_z_score) > 3.5][col]
    outliers_mod_z[col] = len(outliers)

print("\nModified Z-Score Outliers:\n", pd.Series(outliers_mod_z))

# ==========================
# STEP 8: DATA TRANSFORMATION
# ==========================
print("\n===== DATA TRANSFORMATION =====")

# 1. Normalization (0-1)
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[np.number])),
                       columns=df.select_dtypes(include=[np.number]).columns)

# 2. Standardization (mean=0, std=1)
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df.select_dtypes(include=[np.number])),
                      columns=df.select_dtypes(include=[np.number]).columns)

# 3. Encoding Categorical Variables
cat_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))

print("✅ Transformations Done: Normalization, Standardization, Label Encoding")

# ==========================
# STEP 9: COMPARISON OF OUTLIER DETECTION METHODS
# ==========================
comparison = pd.DataFrame({
    'IQR_Outliers': pd.Series(outliers_iqr),
    'ZScore_Outliers': pd.Series(outliers_zscore),
    'ModifiedZScore_Outliers': pd.Series(outliers_mod_z)
})
print("\n===== COMPARISON OF OUTLIER METHODS =====")
print(comparison)

# =====================================
# Step 10: DATA SMOOTHING
# =====================================
print("\n===== DATA SMOOTHING =====")

numeric_cols = df.select_dtypes(include=[np.number]).columns

# 1. Moving Average
window_size = 3
df_ma = df[numeric_cols].rolling(window=window_size, min_periods=1).mean()
print("\nMoving Average (window=3):\n", df_ma.head())

# 2. Simple Binning
bins = 3
df_binned = df.copy()
for col in numeric_cols:
    df_binned[col + "_binned"] = pd.cut(df[col], bins=bins, labels=False)
print("\nSimple Binning:\n", df_binned.head())

# 3. Weighted Moving Average
weights = np.arange(1, window_size + 1)
df_wma = df[numeric_cols].rolling(window=window_size, min_periods=1).apply(lambda x: np.dot(x, weights[-len(x):])/weights[-len(x):].sum(), raw=True)
print("\nWeighted Moving Average:\n", df_wma.head())

# =====================================
# Step 11: ADDITIONAL NORMALIZATION (Decimal Scaling)
# =====================================
print("\n===== DECIMAL SCALING NORMALIZATION =====")
df_decimal_scaled = df[numeric_cols].copy()
for col in numeric_cols:
    max_val = df[col].abs().max()
    j = len(str(int(max_val)))
    df_decimal_scaled[col] = df[col] / (10**j)
print(df_decimal_scaled.head())

# =====================================
# Step 12: REDUNDANCY ANALYSIS
# =====================================
print("\n===== REDUNDANCY ANALYSIS =====")

# 1. Pearson Correlation (numeric)
print("\nPearson Correlation (numeric attributes):")
pearson_corr = df[numeric_cols].corr(method='pearson')
print(pearson_corr)

# 2. Chi-Square Test (categorical)
from scipy.stats import chi2_contingency
cat_cols = df.select_dtypes(include=['object', 'category']).columns

if len(cat_cols) >= 2:
    print("\nChi-Square Test (categorical attributes):")
    chi_square_results = {}
    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            table = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
            chi2, p, dof, ex = chi2_contingency(table)
            chi_square_results[f"{cat_cols[i]} vs {cat_cols[j]}"] = p
    print(pd.Series(chi_square_results))
else:
    print("Not enough categorical attributes for Chi-Square Test")

# =====================================
# Step 13: DISCRETIZATION BY INTUITIVE PARTITIONING
# =====================================
print("\n===== DISCRETIZATION BY INTUITIVE PARTITIONING =====")
df_discretized = df.copy()

# Example: for numeric columns, manually define bins based on domain knowledge or intuition
# Here, we use quartiles as a simple intuitive approach
for col in numeric_cols:
    df_discretized[col + "_discretized"] = pd.qcut(df[col], q=4, labels=False)
print(df_discretized.head())


`);
}
