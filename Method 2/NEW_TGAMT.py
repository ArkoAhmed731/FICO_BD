import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM, s
from functools import reduce
import operator
from sklearn.model_selection import train_test_split
import pickle
import os


# =====================================
# 1. Load Training Data and Split
# =====================================
train_path = r"Dataset for method 2\data_train.csv"
df = pd.read_csv(train_path)

# Split the data before any preprocessing
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {len(df_train)}, Test set size: {len(df_test)}")

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
    "Good": 850,
    "Standard": 600
}

df_train["Credit_Score_Numeric"] = df_train["Credit_Score"].map(mapping)
df_train = df_train.dropna(subset=["Credit_Score_Numeric"])
target_column = "Credit_Score_Numeric"
print("Target value counts after mapping:")
print(df_train[target_column].value_counts())
print(f"Total rows with numerical credit scores: {len(df_train)}")
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
# 8. Calculate Accuracy Metrics
# =====================================
if "Credit_Score" in df_test.columns:
    # Define the score ranges for classification
    def get_credit_level(score):
        if 350 <= score <= 599:
            return "Poor"
        elif 600 <= score <= 700:
            return "Standard"
        else:
            return "Good"
    
    # Convert numeric predictions to credit levels
    predicted_levels = [get_credit_level(score) for score in predictions]
    actual_levels = df_test["Credit_Score"].values
    
    # Calculate accuracy
    correct_predictions = sum(p == a for p, a in zip(predicted_levels, actual_levels))
    accuracy = (correct_predictions / len(actual_levels)) * 100
    
    print("\nAccuracy Metrics:")
    print(f"Exact Level Match Accuracy: {accuracy:.2f}%")
    
# =====================================
# 9. Save Final Output
# =====================================
df_test["tgam_score"] = predictions
df_test["customer_id"] = ['C' + str(i).zfill(5) for i in range(1, len(df_test) + 1)]
final_df = df_test[["customer_id", "tgam_score"]]
final_df.to_csv("credit_scores_asfi.csv", index=False)
print("\nFinal TGAMT output saved to credit_scores_asfi.csv")

# =====================================
# 10. Save Model Components with Pickle
# =====================================
model_components = {
    'tree': tree,
    'leaf_gams': leaf_gams,
    'imputer': imputer,
    'numeric_features': numeric_features,
    'unique_leaves': unique_leaves
}

# Create absolute path for models directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
model_path = os.path.join(models_dir, "tgamt_model.pkl")

try:
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory created/verified at: {models_dir}")

    # Save the model components
    with open(model_path, "wb") as f:
        pickle.dump(model_components, f)
    
    print(f"Model components successfully saved to: {model_path}")
    
except Exception as e:
    print(f"Error saving model: {str(e)}")
    print(f"Attempted to save to: {model_path}")
