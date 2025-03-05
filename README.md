Custom Transparent Generalized Additive Model Tree (TGAMT) for Credit Scoring

## Abstract

This paper presents a custom implementation of a Transparent Generalized Additive Model Tree (TGAMT) for credit scoring. Two approaches are compared: one that includes an explicit binning step for numeric features and another that relies solely on the decision tree's intrinsic segmentation. We discuss how each method partitions the data, fits local generalized additive models (GAMs), and produces interpretable credit score predictions in the range [350, 850]. Our findings suggest that explicit binning may be redundant when a decision tree is used, as the tree already segments the data adaptively.

## 1. Introduction

In regulated financial environments, credit scoring models must balance predictive accuracy with transparency and interpretability. Traditional black-box models often lack the explainability required by regulators, whereas fully transparent models may not capture complex relationships. The TGAMT approach aims to bridge this gap by combining the interpretability of generalized additive models (GAMs) with the adaptive segmentation capabilities of decision trees. This paper describes our custom TGAMT implementation and examines two variants: one with an explicit binning step for the numeric features and one without it.

## 2. Methodology

### 2.1 Data Preprocessing and Feature Cleaning

Our dataset contains key numeric features (such as Age, Annual Income, Payment Delays, etc.) and a categorical target ("Credit_Score") with ratings like "Poor," "Fair," "Good," "Very Good," and "Excellent." In both variants, we perform the following steps:
- **Cleaning:** Remove non-numeric characters (e.g., underscores) from numeric columns and convert them to numeric types.
- **Imputation:** Replace missing values with the median value calculated from the training data.
- **Target Conversion:** Map the categorical credit score to a numeric value using the mapping:
  - "Poor" → 350  
  - "Fair" → 550  
  - "Good" → 650  
  - "Very Good" → 720  
  - "Excellent" → 850

### 2.2 Variant 1: TGAMT with Explicit Binning

In this variant, we apply an explicit binning step using `KBinsDiscretizer` to convert continuous numeric features into discrete bins (e.g., 10 quantile-based bins). The rationale is that explicit binning can reduce noise and outlier influence by smoothing the data before segmentation. The binned data is then used as the input to a decision tree.

- **Binning:**  
  Each numeric feature is binned into a fixed number of bins. This process reduces the data’s variability, potentially increasing robustness.
  
- **Decision Tree Partitioning:**  
  A DecisionTreeRegressor (with a maximum depth, e.g., 3) is trained on the binned numeric features to partition the training data into homogeneous leaves. Each leaf corresponds to a segment of the data that shares similar characteristics.
  
- **Local GAM Fitting:**  
  For each leaf node, a local GAM is fitted using pyGAM’s LinearGAM. This GAM uses spline functions to model the (potentially non-linear) relationship between the features (in their binned form) and the credit score within that segment.

### 2.3 Variant 2: TGAMT without Explicit Binning

In the second variant, explicit binning is omitted. The decision tree is applied directly to the continuous numeric features. Since decision trees inherently split the data based on feature thresholds, they perform an adaptive form of binning. Thus, explicit binning may be redundant in this context.

- **Direct Tree Partitioning:**  
  The decision tree partitions the continuous data directly. This adaptive segmentation may preserve more detailed information than fixed binning.
  
- **Local GAM Fitting:**  
  As with Variant 1, local GAMs are fitted within each leaf using the continuous numeric features. The GAMs learn the relationship between the features and the credit score in that specific partition.

### 2.4 Calibration and Prediction

Regardless of the variant, once local GAMs are trained:
- **Test Data Processing:**  
  The decision tree assigns each test sample to a corresponding leaf, and the local GAM for that leaf predicts a raw credit score.
  
- **Post-Processing:**  
  Predictions are clipped to ensure they lie within the traditional credit score range [350, 850] and then rounded.
  
- **Calibration (Optional):**  
  If systematic bias is detected (e.g., predictions consistently lower than the actual target distribution), a calibration step using a simple linear regression may be applied to adjust the predictions.

## 3. Discussion

### 3.1 Benefits of the TGAMT Approach

- **Local Adaptation:**  
  By partitioning the data, TGAMT can capture regional variations in the relationship between features and credit scores. This is especially useful if different customer segments exhibit different patterns.

- **Transparency:**  
  Both the decision tree segmentation and the additive GAM structure are highly interpretable. Stakeholders can examine how data is segmented and how individual features contribute to the credit score in each segment.

- **Flexibility:**  
  The approach allows for the integration of a calibration step, ensuring that the final predictions align well with traditional credit score ranges.

### 3.2 Comparison: With vs. Without Explicit Binning

- **With Explicit Binning:**  
  - **Pros:**  
    - Smoothing the data can reduce noise and outlier effects.
    - Fixed bins provide a uniform discretization across features.
  - **Cons:**  
    - Potential loss of detailed information due to over-smoothing.
    - May be redundant if the decision tree already effectively segments the data.
    
- **Without Explicit Binning:**  
  - **Pros:**  
    - Preserves the original continuous nature of the data, allowing the decision tree to create adaptive splits.
    - May capture finer variations in the data.
  - **Cons:**  
    - If the data is very noisy, direct splits may lead to overfitting or unstable partitions.
    
Given that decision trees naturally partition continuous features, explicit binning might not always be necessary. In many cases, the tree’s adaptive binning is sufficient, and omitting explicit binning may retain more useful information for the local GAMs.

### 3.3 Model Improvement Considerations

- **Hyperparameter Tuning:**  
  Adjust the maximum depth of the decision tree or the number of bins (if using explicit binning) to optimize the balance between bias and variance.
  
- **Calibration:**  
  Incorporate a calibration model to correct any systematic bias in the TGAMT predictions.
  
- **Feature Engineering:**  
  Consider adding interaction terms or additional features to further improve the model’s predictive power.
  
- **Comparative Evaluation:**  
  Experiment with both variants (with and without explicit binning) on validation data to assess which approach yields more accurate and stable credit score predictions.

## 4. Conclusion

The custom TGAMT approach presented in this paper combines decision tree segmentation with local GAM fitting to produce transparent, interpretable credit scores. Two variants were discussed: one that applies explicit binning to reduce noise, and one that leverages the inherent adaptive binning of decision trees. While explicit binning can help in cases of high noise, it may also remove useful detail. The choice between the two methods should be guided by the characteristics of the data. Calibration and hyperparameter tuning are crucial to ensure that the final credit scores align with traditional ranges and accurately reflect credit risk.






## Running the Project

### Setup and Installation
1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Start the FastAPI backend:
```bash
uvicorn api:app --reload
```
The API server will start at http://localhost:8000, docs http://localhost:8000/docs

2. For the webapp:
   - Simply open `/webapp/index.html` in a web browser
   - No additional server is needed as it's a static HTML file

### Accessing the Application
1. Open your web browser and navigate to the `index.html` file
2. Fill in the credit scoring parameters:
   - Age
   - Annual Income
   - Delay from due date
   - Number of Delayed Payments
   - Number of Credit Inquiries
   - Outstanding Debt
   - Total EMI per month
   - Credit Age (years)
3. Click "Calculate Score" to get your credit score prediction


Note: Ensure the FastAPI backend is running before using the webapp.
