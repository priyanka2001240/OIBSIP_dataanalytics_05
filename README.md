# OIBSIP_dataanalytics_05
Task 5: High-Stakes Credit Card Fraud Detection
Internship

Oasis Infobyte – Data Science Internship (Virtual)

Project Goal

The goal of this project was to design and optimize a Credit Card Fraud Detection System capable of identifying fraudulent transactions in highly imbalanced datasets.
The focus was on maximizing Precision and Recall for the minority class to reduce financial loss and ensure reliable fraud detection in real-world applications.

Dataset

The project used a Credit Card Transactions Dataset, where fraudulent transactions account for less than 1% of total records.
This severe class imbalance made the problem both challenging and realistic, requiring specialized modeling and evaluation strategies.

Key Steps & Methodology

Data Understanding & Preprocessing:

Conducted initial EDA to understand class distribution and transaction behavior.

Normalized and scaled numerical features to ensure stable model performance.

Separated features and labels for clean training workflows.

Imbalance Mitigation:

Applied class_weight='balanced' to adjust learning for minority classes.

Experimented with Undersampling to reduce data skew and improve learning on fraud cases.

Ensured no data leakage during preprocessing.

Model Building & Benchmarking:

Trained and compared four classification models:

Random Forest Classifier 🌲

Logistic Regression

Decision Tree Classifier

MLP Neural Network (Multi-Layer Perceptron)

Identified Random Forest as the most robust model with superior Recall and Precision for fraud cases.

Evaluation & Optimization:

Focused on imbalance-sensitive metrics: ROC AUC, Precision-Recall Curves, and AP Score.

Generated Classification Reports to assess fraud vs. non-fraud class performance.

Applied Threshold Tuning to optimize the F1 Score and reduce false negatives.

Deployment Preparation:

Finalized the best-performing Random Forest model.

Serialized both the model and the scaler using Joblib for production-ready deployment.

Tools and Libraries

Python

Pandas & NumPy: Data preprocessing and handling.

Matplotlib & Seaborn: Exploratory analysis and visualization.

Scikit-learn (sklearn): RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, MLPClassifier, train_test_split, precision_recall_curve, roc_auc_score.

Joblib: Model saving and deployment preparation.

Actionable Conclusions & Model Outcome

Random Forest delivered the highest Recall and balanced Precision, making it the most suitable for high-risk fraud detection.

Threshold tuning significantly improved fraud detection rates with minimal increase in false alarms.

The project provided a business-ready fraud detection model capable of enhancing financial security and minimizing loss.

File Structure

credit_card_fraud_detection.py – Full ML pipeline and model optimization script.

creditcard.csv – Dataset containing transaction records.

fraud_rf_model.joblib – Saved Random Forest model.

scaler.joblib – Saved scaler object for deployment.

README.md – Documentation file (this file).
