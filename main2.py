import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. DATA SEEDING (Proper Automated DataFrame Construction) ---
np.random.seed(42)
n_samples = 100

# Generating automated continuous matrices between 0.0 and 1.0
data = {
    'Age': np.random.uniform(0.0, 1.0, n_samples),
    'Math_Score': np.random.uniform(0.0, 1.0, n_samples),
    'Stream_CS': np.random.choice([0, 1], n_samples),
    'Stream_EC': np.random.choice([0, 1], n_samples),
    'Stream_ME': np.random.choice([0, 1], n_samples),
}

df = pd.DataFrame(data)

# Creating a mathematical rule for the target feature: High math scores lead to placement
df['Placed'] = (df['Math_Score'] > 0.5).astype(int)

# Isolating independent variables (X) from dependent variable (y)
X = df.drop(columns=['Placed'])
y = df['Placed']

# Standard 70/30 split allocation for tracking training bias
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. LOGISTIC REGRESSION ---
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# --- 3. DECISION TREE CLASSIFIER ---
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# --- 4. STRUCTURAL EVALUATION SUMMARY ---
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic Regression': [
        accuracy_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_log)
    ],
    'Decision Tree': [
        accuracy_score(y_test, y_pred_tree),
        precision_score(y_test, y_pred_tree),
        recall_score(y_test, y_pred_tree),
        f1_score(y_test, y_pred_tree)
    ]
}

print("\n--- SUPERVISED MODELS EVALUATION MATRICES ---")
print(pd.DataFrame(metrics).to_string(index=False))
