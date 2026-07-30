# predictiive_matainance_model
# Predictive Maintenance for Industrial Equipment

A machine learning project that predicts machine failure using industrial sensor data, built to help plan maintenance before a breakdown happens rather than after.

 Dataset

- **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
- 10,000 rows of industrial machine sensor readings
- Features: air temperature, process temperature, rotational speed, torque, tool wear, product quality type
- Target: Machine failure (binary) — only 339 failures (3.39%), a realistic and heavily imbalanced rare-event problem

Approach

1. **EDA** — checked for missing values (none) and confirmed the class imbalance
2. **Feature engineering** — created derived features that capture real mechanical stress signals:
   - temp_diff — process temperature minus air temperature (heat stress)
   - power — torque × rotational speed (mechanical effort)
   - wear_torque_ratio — tool wear relative to torque (worn tool under load)
3. **Train/test split** — 80/20, stratified to preserve the failure ratio in both sets
4. **Scaling** — StandardScaler applied to all features
5. **Handling imbalance** — tested SMOTE at various strengths and scale_pos_weight; found that heavy oversampling caused overfitting, informing the final approach
6. **Model selection** — used GridSearchCV (5-fold cross-validation) to search across tree depth, number of estimators, learning rate, and class weighting, rather than manually tuning by trial and error
7. **Evaluation** — compared train vs. test performance directly to catch and correct overfitting before finalizing the model

 Results

Final model: XGBoost Classifier, selected via GridSearchCV

Metric  Test Set 
 F1 Score 0.828 
 ROC-AUC  0.982 
 False alarms  7 / 2000 
 Missed failures  15 / 68 real failures 

The model prioritizes catching real failures (recall) over minimizing false alarms, since missed failures are costlier in a real maintenance setting than an unnecessary inspection.

 Files

- Predictive_Maintainance_For_Industrial_Equipment.ipynb — full notebook (EDA, feature engineering, model training, evaluation)
- predictive_maintainance_model.pkl — trained XGBoost model
- scaler.pkl — fitted StandardScaler used to preprocess new input data

 #Tech Stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn — built and run in Google Colab


Author Chukwu Chioma Excellent
