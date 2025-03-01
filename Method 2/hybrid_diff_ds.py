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
    "Credit_Age_years"  # May be missing; will be filtered out if not present.
]

desired_categorical_features = [
    "Credit_Mix",
    "Payment_Behaviour",
    "Payment_of_Min_Amount_NM",
    "Payment_of_Min_Amount_No",
    "Payment_of_Min_Amount_Yes"
]

# Filter features to those present in training data
numeric_features = [col for col in desired_numeric_features if col in df_train.columns]
categorical_features = [col for col in desired_categorical_features if col in df_train.columns]

print("Using numeric features:", numeric_features)
print("Using categorical features:", categorical_features)

# =====================================
# 3. Clean Numeric Columns in Both Datasets
# =====================================
# Remove unwanted characters (e.g., underscores) and convert to numeric.
for col in numeric_features:
    df_train[col] = pd.to_numeric(df_train[col].astype(str).str.replace('_', ''), errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace('_', ''), errors='coerce')

# Impute missing numeric values using median from the training set.
for col in numeric_features:
    median_val = df_train[col].median()
    df_train[col] = df_train[col].fillna(median_val)
    df_test[col] = df_test[col].fillna(median_val)

print("Missing values in training numeric data:")
print(df_train[numeric_features].isnull().sum())

# =====================================
# 4. Compute Unsupervised Credit Score on Training Data
# =====================================
# Define feature directions.
feature_direction = {
    "Age": "positive",
    "Annual_Income": "positive",
    "Delay_from_due_date": "negative",
    "Num_of_Delayed_Payment": "negative",
    "Num_Credit_Inquiries": "negative",
    "Outstanding_Debt": "negative",
    "Total_EMI_per_month": "negative",
    "Credit_Age_years": "positive"
}
# Filter to numeric_features available.
feature_direction = {k: v for k, v in feature_direction.items() if k in numeric_features}

# Fit KBinsDiscretizer on training numeric features.
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
bins_array_train = est.fit_transform(df_train[numeric_features])
bins_df_train = pd.DataFrame(bins_array_train, columns=numeric_features)

# Function to compute feature score.
def compute_feature_score(bin_value, direction):
    if direction == "positive":
        return bin_value + 1  # converts 0->1, 4->5
    else:
        return 5 - bin_value  # converts 0->5, 4->1

# Compute scores for each numeric feature.
for feature in numeric_features:
    bins_df_train[feature + "_score"] = bins_df_train[feature].apply(lambda x: compute_feature_score(x, feature_direction[feature]))

score_columns = [f + "_score" for f in numeric_features]
df_train['raw_score'] = bins_df_train[score_columns].sum(axis=1)

min_possible = len(numeric_features) * 1
max_possible = len(numeric_features) * 5

df_train['unsupervised_score'] = 350 + ((df_train['raw_score'] - min_possible) / (max_possible - min_possible)) * 500
df_train['unsupervised_score'] = df_train['unsupervised_score'].round().astype(int)

# =====================================
# 5. Compute Unsupervised Credit Score on Testing Data
# =====================================
bins_array_test = est.transform(df_test[numeric_features])
bins_df_test = pd.DataFrame(bins_array_test, columns=numeric_features)

for feature in numeric_features:
    bins_df_test[feature + "_score"] = bins_df_test[feature].apply(lambda x: compute_feature_score(x, feature_direction[feature]))

score_columns = [f + "_score" for f in numeric_features]
df_test['raw_score'] = bins_df_test[score_columns].sum(axis=1)
df_test['unsupervised_score'] = 350 + ((df_test['raw_score'] - min_possible) / (max_possible - min_possible)) * 500
df_test['unsupervised_score'] = df_test['unsupervised_score'].round().astype(int)

# =====================================
# 6. Assign Unique Customer IDs to Test Data
# =====================================
df_test['customer_id'] = ['C' + str(i).zfill(5) for i in range(1, len(df_test) + 1)]

# =====================================
# 7. Supervised Modeling: Use Unsupervised Score as Target
# =====================================
df_train['target_score'] = df_train['unsupervised_score']

X_train = df_train[numeric_features + categorical_features]
y_train = df_train['target_score']
X_test = df_test[numeric_features + categorical_features]

# Build preprocessing pipelines for supervised learning with imputation.
num_pipeline_supervised = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('binner', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
cat_pipeline_supervised = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor_supervised = ColumnTransformer(
    transformers=[
        ('num', num_pipeline_supervised, numeric_features),
        ('cat', cat_pipeline_supervised, categorical_features)
    ]
)

# Build supervised regression pipelines.
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor_supervised),
    ('regressor', LinearRegression())
])
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor_supervised),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the models.
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# =====================================
# 8. Predict Supervised Scores on Test Data
# =====================================
df_test['lr_score'] = pipeline_lr.predict(X_test).round().astype(int)
df_test['rf_score'] = pipeline_rf.predict(X_test).round().astype(int)

# =====================================
# 9. Prepare Final Output CSV
# =====================================
final_df = df_test[['customer_id', 'unsupervised_score', 'lr_score', 'rf_score']]
final_df.to_csv("final_credit_scores_unsupervised_LR_RF.csv", index=False)
print("Final output saved to final_credit_scores_unsupervised_LR_RF.csv")
