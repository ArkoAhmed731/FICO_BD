import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# =====================================
# 1. Load the Training and Testing Data
# =====================================
train_path = r"D:\Credit Score\SimpliCredit\Dataset for method 2\data_train.csv"
test_path = r"D:\Credit Score\SimpliCredit\Dataset for method 2\data_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# =====================================
# 2. Define Desired Numeric and Categorical Features
# =====================================
desired_numeric_features = [
    "Age",
    "Annual_Income",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "Credit_Age_years"  # Optional; will be filtered if missing.
]

desired_categorical_features = [
    "Credit_Mix",
    "Payment_Behaviour",
    "Payment_of_Min_Amount_NM",
    "Payment_of_Min_Amount_No",
    "Payment_of_Min_Amount_Yes"
]

# Filter features to those present in the training data.
numeric_features = [col for col in desired_numeric_features if col in df_train.columns]
categorical_features = [col for col in desired_categorical_features if col in df_train.columns]

print("Using numeric features:", numeric_features)
print("Using categorical features:", categorical_features)

# =====================================
# 3. Clean Numeric Columns in Both Datasets
# =====================================
for col in numeric_features:
    df_train[col] = pd.to_numeric(df_train[col].astype(str).str.replace('_', ''), errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace('_', ''), errors='coerce')

for col in numeric_features:
    median_val = df_train[col].median()
    df_train[col] = df_train[col].fillna(median_val)
    df_test[col] = df_test[col].fillna(median_val)

print("Missing values in training numeric data:")
print(df_train[numeric_features].isnull().sum())

# =====================================
# 4. Convert the Target Column to Numeric
# =====================================
# Assume the target column is "Credit_Score" (non-numerical)
# Map the categorical ratings to numeric values using the updated mapping.
mapping = {
    "Poor": 400,
    "Fair": 550,
    "Good": 650,
    "Very Good": 720,
    "Excellent": 810
}

df_train["Credit_Score_Numeric"] = df_train["Credit_Score"].map(mapping)

# Drop rows where mapping failed (i.e., target is NaN)
df_train = df_train.dropna(subset=["Credit_Score_Numeric"])

target_column = "Credit_Score_Numeric"
print("Target value counts after mapping:")
print(df_train[target_column].value_counts())

X_train = df_train[numeric_features + categorical_features]
y_train = df_train[target_column]
X_test = df_test[numeric_features + categorical_features]

# =====================================
# 5. Build Preprocessing Pipelines for Supervised Learning
# =====================================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('binner', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ]
)

# =====================================
# 6. Build and Train Supervised Regression Models
# =====================================
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# =====================================
# 7. Predict on the Test Data
# =====================================
df_test['lr_score'] = pipeline_lr.predict(X_test).round().astype(int)
df_test['rf_score'] = pipeline_rf.predict(X_test).round().astype(int)

# =====================================
# 8. Assign Unique Customer IDs to Test Data
# =====================================
df_test['customer_id'] = ['C' + str(i).zfill(5) for i in range(1, len(df_test) + 1)]

# =====================================
# 9. Prepare Final Output CSV
# =====================================
final_df = df_test[['customer_id', 'lr_score', 'rf_score']]
final_df.to_csv("final_credit_scores_supervised.csv", index=False)
print("Final output saved to final_credit_scores_supervised.csv")
