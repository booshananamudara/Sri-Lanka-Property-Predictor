"""
Sri Lanka Property Price Predictor
==================================
Complete ML Pipeline: Preprocessing -> Training -> Evaluation -> XAI

Dataset: data/house_prices.csv (15,327 listings from ikman.lk)
Model: CatBoost Regressor
Output: models/catboost_model.pkl, models/preprocessor.pkl
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# For XAI
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

RAW_DATA = os.path.join(DATA_DIR, "house_prices.csv")
CLEANED_DATA = os.path.join(DATA_DIR, "cleaned_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING
# ============================================================

print("=" * 60)
print("STEP 1: Loading Raw Data")
print("=" * 60)

df = pd.read_csv(RAW_DATA)
print(f"  Raw dataset shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Data Cleaning & Feature Engineering")
print("=" * 60)

# --- 2a. Parse Price (target variable) ---
def parse_price(price_str):
    """Convert 'Rs 5,400,000' to float 5400000.0"""
    if pd.isna(price_str):
        return np.nan
    cleaned = str(price_str)
    cleaned = re.sub(r'^Rs\.?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(',', '').strip()
    match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if match:
        return float(match.group(1))
    return np.nan

df['Price_LKR'] = df['Price'].apply(parse_price)
print(f"  Parsed prices: {df['Price_LKR'].notna().sum()} valid out of {len(df)}")

# --- 2b. Parse Bedrooms ---
def parse_beds(val):
    """Convert '3' or '10+' to int"""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == '10+':
        return 10
    match = re.search(r'(\d+)', val)
    return int(match.group(1)) if match else np.nan

df['Bedrooms'] = df['Beds'].apply(parse_beds)

# --- 2c. Parse Bathrooms ---
df['Bathrooms'] = df['Baths'].apply(parse_beds)  # Same logic as beds

# --- 2d. Parse Land Size (perches) ---
def parse_land_size(val):
    """Convert '50.0 perches' to float 50.0"""
    if pd.isna(val):
        return np.nan
    val = str(val)
    match = re.search(r'([\d,]+(?:\.\d+)?)\s*(?:perch|perches|p)', val, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    return np.nan

df['Land_Size_Perches'] = df['Land size'].apply(parse_land_size)

# --- 2e. Parse House Size (sqft) ---
def parse_house_size(val):
    """Convert '1,600.0 sqft' to float 1600.0"""
    if pd.isna(val):
        return np.nan
    val = str(val)
    match = re.search(r'([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)', val, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    return np.nan

df['House_Size_Sqft'] = df['House size'].apply(parse_house_size)

# --- 2f. Parse Location ---
def parse_city(loc):
    """Extract city from ' Matara City,  Matara' -> 'Matara City'"""
    if pd.isna(loc):
        return 'Unknown'
    parts = str(loc).split(',')
    return parts[0].strip() if parts else 'Unknown'

def parse_district(loc):
    """Extract district from ' Matara City,  Matara' -> 'Matara'"""
    if pd.isna(loc):
        return 'Unknown'
    parts = str(loc).split(',')
    return parts[-1].strip() if len(parts) > 1 else 'Unknown'

df['City'] = df['Location'].apply(parse_city)
df['District'] = df['Location'].apply(parse_district)

print(f"  Unique cities: {df['City'].nunique()}")
print(f"  Unique districts: {df['District'].nunique()}")

# --- 2g. Drop rows with missing key values ---
print(f"\n  Before cleaning: {len(df)} rows")

# Drop rows where Price is missing or zero
df = df[df['Price_LKR'] > 0].copy()
print(f"  After removing zero/missing prices: {len(df)}")

# Drop rows where key features are missing
df = df.dropna(subset=['Bedrooms', 'Bathrooms', 'Land_Size_Perches', 'House_Size_Sqft'])
print(f"  After removing missing features: {len(df)}")

# --- 2h. Remove Outliers ---
# Price outliers (remove extremes)
q1 = df['Price_LKR'].quantile(0.01)
q99 = df['Price_LKR'].quantile(0.99)
df = df[(df['Price_LKR'] >= q1) & (df['Price_LKR'] <= q99)]
print(f"  After removing price outliers (1%-99%): {len(df)}")

# Land size outliers (3000 perches is clearly a typo)
df = df[df['Land_Size_Perches'] <= 500]
print(f"  After removing land size outliers (>500p): {len(df)}")

# House size outliers
df = df[df['House_Size_Sqft'] <= 20000]
print(f"  After removing house size outliers (>20000sqft): {len(df)}")

# Bedrooms & Bathrooms sanity check
df = df[(df['Bedrooms'] >= 1) & (df['Bedrooms'] <= 10)]
df = df[(df['Bathrooms'] >= 1) & (df['Bathrooms'] <= 10)]
print(f"  After bedroom/bathroom filter (1-10): {len(df)}")

# --- 2i. Select Features for Model ---
feature_columns = ['Bedrooms', 'Bathrooms', 'Land_Size_Perches', 'House_Size_Sqft', 'City', 'District']
target_column = 'Price_LKR'

df_model = df[feature_columns + [target_column]].copy()
df_model = df_model.reset_index(drop=True)

print(f"\n  Final dataset shape: {df_model.shape}")
print(f"  Features: {feature_columns}")
print(f"  Target: {target_column}")

# --- 2j. Encode Categorical Features ---
label_encoders = {}
categorical_cols = ['City', 'District']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

# Save cleaned data
df_model.to_csv(CLEANED_DATA, index=False)
print(f"\n  Saved cleaned data to: {CLEANED_DATA}")

# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Train-Test Split")
print("=" * 60)

X = df_model[feature_columns]
y = df_model[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# ============================================================
# STEP 4: MODEL TRAINING (CatBoost Regressor)
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Training CatBoost Regressor")
print("=" * 60)

# CatBoost handles categorical features natively
cat_features_indices = [feature_columns.index(c) for c in categorical_cols]

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    eval_metric='RMSE',
    early_stopping_rounds=50,
    cat_features=cat_features_indices,
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=100
)

print("\n  Training complete!")

# ============================================================
# STEP 5: MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Model Evaluation")
print("=" * 60)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

# Test metrics
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n  {'Metric':<20} {'Training':>15} {'Testing':>15}")
print(f"  {'-'*50}")
print(f"  {'MAE (LKR)':<20} {train_mae:>15,.0f} {test_mae:>15,.0f}")
print(f"  {'RMSE (LKR)':<20} {train_rmse:>15,.0f} {test_rmse:>15,.0f}")
print(f"  {'R-Squared':<20} {train_r2:>15.4f} {test_r2:>15.4f}")

print(f"\n  In millions LKR:")
print(f"  {'MAE':<20} {train_mae/1e6:>15.2f}M {test_mae/1e6:>15.2f}M")
print(f"  {'RMSE':<20} {train_rmse/1e6:>15.2f}M {test_rmse/1e6:>15.2f}M")

# ============================================================
# STEP 6: SAVE MODEL & PREPROCESSOR
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Saving Model & Preprocessor")
print("=" * 60)

# Save model as .pkl
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"  Model saved to: {MODEL_PATH}")

# Save preprocessor (label encoders + feature columns)
preprocessor = {
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'categorical_cols': categorical_cols,
    'target_column': target_column,
}
with open(PREPROCESSOR_PATH, 'wb') as f:
    pickle.dump(preprocessor, f)
print(f"  Preprocessor saved to: {PREPROCESSOR_PATH}")

# ============================================================
# STEP 7: EXPLAINABLE AI (XAI)
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Explainable AI (XAI)")
print("=" * 60)

# --- 7a. Feature Importance (built-in CatBoost) ---
print("\n  [7a] CatBoost Feature Importance:")
feature_importances = model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '#' * int(row['Importance'] / 2)
    print(f"      {row['Feature']:<25} {row['Importance']:>8.2f}  {bar}")

# Plot Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
ax.set_title('CatBoost Feature Importance', fontsize=16, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
feat_imp_path = os.path.join(NOTEBOOK_DIR, 'feature_importance.png')
plt.savefig(feat_imp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {feat_imp_path}")

# --- 7b. SHAP Analysis ---
print("\n  [7b] SHAP Analysis (this may take a moment)...")

# Use a subset for SHAP to keep it fast
X_shap = X_test.sample(min(500, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# SHAP Summary Plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, feature_names=feature_columns, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Price', fontsize=14, fontweight='bold')
plt.tight_layout()
shap_summary_path = os.path.join(NOTEBOOK_DIR, 'shap_summary.png')
plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {shap_summary_path}")

# SHAP Bar Plot (Mean absolute SHAP values)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, feature_names=feature_columns, plot_type='bar', show=False)
plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
plt.tight_layout()
shap_bar_path = os.path.join(NOTEBOOK_DIR, 'shap_bar.png')
plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {shap_bar_path}")

# --- 7c. Actual vs Predicted Plot ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test / 1e6, y_pred_test / 1e6, alpha=0.3, s=10, color='#4285f4')
max_val = max(y_test.max(), max(y_pred_test)) / 1e6
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Price (Million LKR)', fontsize=12)
ax.set_ylabel('Predicted Price (Million LKR)', fontsize=12)
ax.set_title(f'Actual vs Predicted Prices (R\u00b2 = {test_r2:.4f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, max_val * 1.05)
ax.set_ylim(0, max_val * 1.05)
plt.tight_layout()
actual_vs_pred_path = os.path.join(NOTEBOOK_DIR, 'actual_vs_predicted.png')
plt.savefig(actual_vs_pred_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {actual_vs_pred_path}")

# --- 7d. Residual Distribution ---
residuals = (y_test - y_pred_test) / 1e6
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(residuals, bins=50, color='#4285f4', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual (Million LKR)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
residual_path = os.path.join(NOTEBOOK_DIR, 'residual_distribution.png')
plt.savefig(residual_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {residual_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"""
  Dataset:        {len(df_model)} records (cleaned from {15327} raw)
  Model:          CatBoost Regressor
  Features:       {feature_columns}
  
  Test R-Squared: {test_r2:.4f}
  Test MAE:       {test_mae/1e6:.2f} Million LKR
  Test RMSE:      {test_rmse/1e6:.2f} Million LKR
  
  Files Saved:
    - Model:        {MODEL_PATH}
    - Preprocessor: {PREPROCESSOR_PATH}
    - Cleaned Data: {CLEANED_DATA}
    - Plots:        {NOTEBOOK_DIR}/
""")
