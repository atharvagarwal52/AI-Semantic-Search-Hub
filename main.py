import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# STEP 1: DATA LOADING (Creating the Dataset)
# ==========================================
raw_data = {
    'Student_ID': [101, 102, 103, 104, 105, 106],
    'Age': [18, 19, np.nan, 20, 19, np.nan],          
    'Stream': ['CS', 'EC', 'CS', np.nan, 'ME', 'CS'],   
    'Math_Score': [85, 90, 78, 92, 88, 95],            
    'Placed': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes']   
}

df = pd.DataFrame(raw_data)
print("--- 1. RAW MESSY DATASET ---")
print(df, "\n" + "="*50 + "\n")

# ==========================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("--- 2. EXPLORATORY DATA ANALYSIS ---")
print(f"Dataset Shape (Rows, Columns): {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum(), "\n" + "="*50 + "\n")

# ==========================================
# STEP 3: HANDLING MISSING VALUES
# ==========================================
# Fill missing 'Age' using the Mean (Average) value
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

# FIXED: Extracting the clean string value [0] from the mode series
mode_stream = df['Stream'].mode()[0]
df['Stream'] = df['Stream'].fillna(mode_stream)

print("--- 3. DATA AFTER HANDLING MISSING VALUES ---")
print(df, "\n" + "="*50 + "\n")

# ==========================================
# STEP 4: ENCODING CATEGORICAL VARIABLES
# ==========================================
df['Placed'] = df['Placed'].map({'Yes': 1, 'No': 0})

# Use One-Hot Encoding for 'Stream'
df = pd.get_dummies(df, columns=['Stream'], dtype=int)

print("--- 4. DATA AFTER ENCODING CATEGORICAL FEATURES ---")
print(df, "\n" + "="*50 + "\n")

# ==========================================
# STEP 5: FEATURE SELECTION
# ==========================================
df = df.drop(columns=['Student_ID'])

print("--- 5. DATA AFTER FEATURE SELECTION ---")
print(df, "\n" + "="*50 + "\n")

# ==========================================
# STEP 6: DATA NORMALIZATION (SCALING)
# ==========================================
scaler = MinMaxScaler()
df[['Age', 'Math_Score']] = scaler.fit_transform(df[['Age', 'Math_Score']])

print("--- 6. FINAL CLEANED & PREPROCESSED DATASET ---")
print(df)
