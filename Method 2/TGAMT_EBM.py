import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split

# =====================================
# 1. Load Training and Testing Data
# =====================================
train_path = r"D:\Credit Score\SimpliCredit\Dataset for method 2\data_train.csv"
test_path = r"D:\Credit Score\SimpliCredit\Dataset for method 2\data_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# =====================================
# 2. Define Desired Features
# =====================================
desired_numeric_features = [
    "Age",
    "Annual_Income",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "Credit_Age_years"  # Optional; if missing, will be excluded.
]

desired_categorical_features = [
    "Credit_Mix",
    "Payment_Behaviour",
    "Payment_of_Min_Amount_NM",
    "Payment_of_Min_Amount_No",
    "Payment_of_Min_Amount_Yes"
]

numeric_features = [col for col in desired_numeric_features if col in df_train.columns]
categorical_features = [col for col in desired_categorical_features if col in df_train.columns]

print("Using numeric features:", numeric_features)
print("Using categorical features:", categorical_features)

# =====================================
# 3. Clean and Impute Numeric Features
# =====================================
# Remove non-numeric characters (e.g., underscores) and convert to float.
for col in numeric_features:
    df_train[col] = pd.to_numeric(df_train[col].astype(str).str.replace('_', ''), errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace('_', ''), errors='coerce')

# Impute missing values using median from training set.
for col in numeric_features:
    median_val = df_train[col].median()
    df_train[col] = df_train[col].fillna(median_val)
    df_test[col] = df_test[col].fillna(median_val)

print("Missing values in training numeric data:")
print(df_train[numeric_features].isnull().sum())

# =====================================
# 4. Convert Target Column to Numeric Score
# =====================================
# Assume the target column is "Credit_Score" (non-numerical)
# Map the values using our desired mapping.
mapping = {
    "Poor": 350,
    "Fair": 550,
    "Good": 650,
    "Very Good": 720,
    "Excellent": 850
}

df_train["Credit_Score_Numeric"] = df_train["Credit_Score"].map(mapping)
df_train = df_train.dropna(subset=["Credit_Score_Numeric"])  # drop rows that could not be mapped
target_column = "Credit_Score_Numeric"
print("Target value counts after mapping:")
print(df_train[target_column].value_counts())

# =====================================
# 5. Prepare Data for Supervised Learning
# =====================================
X_train = df_train[numeric_features + categorical_features]
y_train = df_train[target_column]
X_test = df_test[numeric_features + categorical_features]

# =====================================
# 6. Build Preprocessing Pipeline
# =====================================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
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
# 7. Build and Train the EBM Model
# =====================================
ebm_model = ExplainableBoostingRegressor(random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ebm', ebm_model)
])

pipeline.fit(X_train, y_train)

# =====================================
# 8. Predict and Post-Process on Test Data
# =====================================
predictions = pipeline.predict(X_test)
# Clip predictions to ensure they lie within [350, 850] and round to the nearest integer.
predictions = np.clip(predictions, 350, 850)
df_test['ebm_score'] = np.round(predictions).astype(int)

# =====================================
# 9. Assign Unique Customer IDs and Save Output
# =====================================
df_test['customer_id'] = ['C' + str(i).zfill(5) for i in range(1, len(df_test) + 1)]
final_df = df_test[['customer_id', 'ebm_score']]
final_df.to_csv("final_credit_scores_ebm.csv", index=False)
print("Final EBM output saved to final_credit_scores_ebm.csv")
