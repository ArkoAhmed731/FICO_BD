import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM, s
from functools import reduce
import operator
from sklearn.model_selection import train_test_split

# =====================================
# 1. Load Training and Testing Data
# =====================================
train_path = r"D:\Credit Score\SimpliCredit\Dataset for method 2\data_train.csv"
test_path = r"D:\Credit Score\SimpliCredit\Dataset for method 2\data_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# =====================================
# 2. Define and Clean Numeric Features
# =====================================
desired_numeric_features = [
    "Age",
    "Annual_Income",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "Credit_Age_years"
]

# Use only those columns that exist in training data.
numeric_features = [col for col in desired_numeric_features if col in df_train.columns]
print("Using numeric features:", numeric_features)

# Remove unwanted characters (e.g., underscores) and convert to numeric.
for col in numeric_features:
    df_train[col] = pd.to_numeric(df_train[col].astype(str).str.replace('_', ''), errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace('_', ''), errors='coerce')

# Impute missing values using the median from the training set.
imputer = SimpleImputer(strategy='median')
df_train[numeric_features] = imputer.fit_transform(df_train[numeric_features])
df_test[numeric_features] = imputer.transform(df_test[numeric_features])

# =====================================
# 3. Convert the Target Column to Numeric (Ordinal)
# =====================================
# Map the target "Credit_Score" using the desired mapping.
mapping = {
    "Poor": 350,
    "Fair": 550,
    "Good": 650,
    "Very Good": 720,
    "Excellent": 850
}
df_train["Credit_Score_Numeric"] = df_train["Credit_Score"].map(mapping)
df_train = df_train.dropna(subset=["Credit_Score_Numeric"])
target_column = "Credit_Score_Numeric"
print("Target value counts after mapping:")
print(df_train[target_column].value_counts())
y_train = df_train[target_column].values

# =====================================
# 4. Recompute Feature Matrix from Filtered Training Data
# =====================================
X_train_numeric = imputer.fit_transform(df_train[numeric_features])

# =====================================
# 5. Fit a Decision Tree to Partition the Training Data
# =====================================
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train_numeric, y_train)
train_leaf_indices = tree.apply(X_train_numeric)

# =====================================
# 6. Fit a GAM on Each Leaf Node Using pyGAM
# =====================================
leaf_gams = {}
unique_leaves = np.unique(train_leaf_indices)
p = X_train_numeric.shape[1]  # number of numeric features

# Use functools.reduce to combine spline terms (starting with the first term).
for leaf in unique_leaves:
    indices = np.where(train_leaf_indices == leaf)[0]
    X_leaf = X_train_numeric[indices]
    y_leaf = y_train[indices]
    # Create a combined TermList by reducing the list with operator.add.
    terms = reduce(operator.add, [s(i) for i in range(p)])
    gam = LinearGAM(terms=terms)
    gam.fit(X_leaf, y_leaf)
    leaf_gams[leaf] = gam

# =====================================
# 7. Predict Using the TGAMT Approach on Test Data
# =====================================
X_test_numeric = imputer.transform(df_test[numeric_features])
test_leaf_indices = tree.apply(X_test_numeric)
predictions = np.zeros(len(X_test_numeric))
for leaf in unique_leaves:
    indices = np.where(test_leaf_indices == leaf)[0]
    if len(indices) > 0:
        X_leaf_test = X_test_numeric[indices]
        predictions[indices] = leaf_gams[leaf].predict(X_leaf_test)

# Clip predictions to [350, 850] and round.
predictions = np.clip(predictions, 350, 850)
predictions = np.round(predictions).astype(int)

# =====================================
# 8. Assign Unique Customer IDs and Save Final Output
# =====================================
df_test["tgam_score"] = predictions
df_test["customer_id"] = ['C' + str(i).zfill(5) for i in range(1, len(df_test) + 1)]
final_df = df_test[["customer_id", "tgam_score"]]
final_df.to_csv("final_credit_scores_tgam.csv", index=False)
print("Final TGAMT output saved to final_credit_scores_tgam.csv")
