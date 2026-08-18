import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# STEP 1: DATA PREPARATION (Continuing Student Dataset)
# ==========================================
np.random.seed(42)
n_samples = 150

# Continuous and scaled features (similar to Week 1 & 2 outputs)
data = {
    'Age': np.random.uniform(0.0, 1.0, n_samples),
    'Math_Score': np.random.uniform(0.0, 1.0, n_samples),
    'Stream_CS': np.random.choice([0, 1], n_samples),
    'Stream_EC': np.random.choice([0, 1], n_samples),
    'Stream_ME': np.random.choice([0, 1], n_samples),
}
X = pd.DataFrame(data)

# Target placement feature based on performance threshold
y = ((X['Math_Score'] * 0.7 + X['Age'] * 0.3) > 0.5).astype(int)

# ==========================================
# STEP 2: CLUSTERING & DIMENSIONALITY REDUCTION
# ==========================================
print("--- 1. UNSUPERVISED CLUSTERING EVALUATION ---")

# Apply PCA for 2D visualization / feature reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA Explained Variance Ratio (2 Components): {pca.explained_variance_ratio_.sum():.2%}\n")

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# Hierarchical (Agglomerative) Clustering
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X)
hierarchical_silhouette = silhouette_score(X, hierarchical_labels)

clustering_summary = pd.DataFrame({
    'Algorithm': ['K-Means Clustering', 'Hierarchical (Ward)'],
    'Clusters (K)': [2, 2],
    'Silhouette Score': [kmeans_silhouette, hierarchical_silhouette]
})
print(clustering_summary.to_string(index=False))

# ==========================================
# STEP 3: HYPERPARAMETER TUNING & CROSS-VALIDATION
# ==========================================
print("\n" + "="*50 + "\n--- 2. MODEL TUNING & CROSS-VALIDATION ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Search space grid
param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5]
}

# Stratified 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Optimal Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# ==========================================
# STEP 4: DETAILED MODEL EVALUATION REPORT
# ==========================================
print("\n" + "="*50 + "\n--- 3. MODEL EVALUATION REPORT ---")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix & Comprehensive Metrics
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:")
print(pd.DataFrame(cm, columns=['Pred: No', 'Pred: Yes'], index=['Actual: No', 'Actual: Yes']))

print(f"\nROC-AUC Score: {roc_auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))