# Ensemble Learning and Decision Trees - Detailed Walkthrough and Analysis

---

## Overview

This project aims to solve both regression and classification problems using various machine learning algorithms. The tasks range from implementing a decision tree regressor by hand to leveraging ensemble methods like bagging, boosting, and stacking for classification. Below, we provide a detailed explanation of each task, the approaches, the outputs, and their analyses.

---

## Contents

1. **Decision Trees (Regression)**
   - Task 1: Implement impurity calculation
   - Task 2: Define a cost function
   - Task 3: Build and train a custom decision tree regressor
   - Task 4: Train regression trees on the Boston dataset
   - Task 5: Optimize hyperparameters using GridSearchCV
   - Task 6: Calculate bias and variance
   - Task 7: Analyze the effect of `min_samples_split` on bias and variance
   - Task 8: Reduce variance with bagging

2. **Classification with Ensemble Models**
   - Task 1: Data preprocessing for the Billionaire dataset
   - Task 2: Train and compare various classifiers
   - Task 3: Hyperparameter tuning for XGBoost
   - Task 4: Train advanced ensemble models (Bagging, Stacking, Voting)
   - Task 5: Identify the best model and report its performance

3. **Analysis and Results**

---

## Decision Trees (Regression)

### Task 1: Implement Impurity Calculation
- **Objective:** Implement a function to calculate impurity using the Mean Absolute Deviation (MAD).
- **Output:**
  - Function was verified using test cases.
  - Results matched expectations, with correct impurity values.

### Task 2: Define a Cost Function
- **Objective:** Implement a cost function to evaluate splits based on impurity.
- **Output:**
  - Cost function was validated on sample datasets.
  - Correct values were computed for split evaluations.

### Task 3: Build and Train a Custom Decision Tree Regressor
- **Objective:** Implement a decision tree regressor from scratch with split selection, tree growth, and prediction logic.
- **Output:**
  - The custom tree predictions aligned closely with scikit-learn's implementation.

### Task 4: Train Regression Trees on the Boston Dataset
- **Objective:** Train decision trees with depths of 1 and 2 and compare their performance.
- **Key Insights:**
  - **Train MAE (depth 1):** 5.09 | **Test MAE:** 4.87
  - **Train MAE (depth 2):** 3.54 | **Test MAE:** 3.66
  - **Overfitting:** Trees with depth 2 did not show significant overfitting.

### Task 5: Optimize Hyperparameters Using GridSearchCV
- **Objective:** Use GridSearchCV to optimize `max_depth` and `min_samples_leaf`.
- **Output:**
  - **Best Parameters:** `{'max_depth': 6, 'min_samples_leaf': 2}`
  - **Test MAE:** 2.46
  - GridSearchCV effectively improved performance by identifying optimal hyperparameters.

### Task 6: Calculate Bias and Variance
- **Objective:** Implement an algorithm to estimate bias and variance using bootstrap sampling.
- **Results:**
  - **Bias²:** 20.21 | **Variance:** 140.41
  - The model exhibited low bias and relatively high variance, characteristic of decision trees.

### Task 7: Analyze Effect of `min_samples_split`
- **Objective:** Plot bias and variance for different `min_samples_split` values.
- **Results:**
  - Increasing `min_samples_split` reduced variance slightly but increased bias.
  - **Key Observation:** A trade-off exists between bias and variance, aligning with theoretical expectations.

### Task 8: Reduce Variance with Bagging
- **Objective:** Use bagging to reduce the variance of the decision tree.
- **Results:**
  - **Bias² (Bagging):** 14.31 | **Variance (Bagging):** 139.28
  - Bagging effectively reduced variance compared to a single tree, validating its theoretical advantage.

---

## Classification with Ensemble Models

### Task 1: Data Preprocessing for the Billionaire Dataset
- **Objective:** Preprocess the dataset for classification tasks by handling missing values and applying transformations.
- **Steps:**
  - Dropped irrelevant columns like `personName`, `city`, etc.
  - Applied one-hot encoding for categorical variables and scaling for numeric variables.
- **Output:** Transformed datasets with:
  - **Train Shape:** (1980, 109)
  - **Test Shape:** (660, 109)

### Task 2: Train and Compare Classifiers
- **Objective:** Train 5 classifiers and compare them using the F1 score.
- **Results:**
  - **Best Model:** Random Forest | **Mean F1:** 0.7881
  - Models like Gradient Boosting and Logistic Regression also performed well, with Decision Tree showing signs of underfitting.

### Task 3: Hyperparameter Tuning for XGBoost
- **Objective:** Optimize XGBoost hyperparameters using GridSearchCV.
- **Results:**
  - **Best Hyperparameters:** `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}`
  - **Test F1 Score:** 0.7693
  - XGBoost outperformed Gradient Boosting but was slightly behind Random Forest.

### Task 4: Train Advanced Ensemble Models
- **Objective:** Train advanced ensembles (Bagging, Voting, Stacking) and compare performance.
- **Results:**
  - **Best Model:** Voting Classifier | **F1 Score:** 0.7787
  - Bagging with Gradient Boosting showed stable performance.
  - Stacking (Logistic Regression) provided competitive results.

### Task 5: Identify the Best Model and Report Performance
- **Best Model:** Voting Classifier
- **Results:**
  - **Test F1 Score:** 0.7787
  - **Test Accuracy:** 0.7909
  - **Classification Report:**
    - Precision (weighted): 0.78
    - Recall (weighted): 0.79
    - F1-Score (weighted): 0.78
  - The model performed well on the majority class but struggled with the minority class.

---

## Key Insights and Conclusions

1. **Regression Tasks:**
   - Decision tree regressor implementations were validated against scikit-learn's models.
   - Bias-variance analysis demonstrated the effectiveness of bagging in reducing variance.

2. **Classification Tasks:**
   - Random Forest and Voting Classifier consistently delivered strong results.
   - Advanced ensembles like stacking provided marginal improvements, emphasizing the utility of diverse models.

3. **Ensemble Learning:**
   - Bagging reduced overfitting for gradient boosting with a large number of trees.
   - Voting combined the strengths of multiple algorithms to achieve robust predictions.

4. **Theoretical Alignment:**
   - All experimental results align with the theoretical expectations of ensemble methods and their impact on bias and variance.
