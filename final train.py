import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import joblib  # For saving and loading the model

# Load the dataset
data = pd.read_parquet('bccc-cpacket-cloud-ddos-2024-merged.parquet')  # Replace with the correct path

# Check for missing values and handle them
numeric_cols = data.select_dtypes(include=[np.number]).columns
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

# Handle null and infinity values in numeric columns
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Handle missing values in non-numeric columns (use mode for categorical features)
for col in non_numeric_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode non-numeric columns using LabelEncoder (for string categories)
for col in non_numeric_cols:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])

# Encode the target label column (assuming the target is named 'label')
label_column = 'label'  # Replace with the actual label column name in your dataset
label_encoder = LabelEncoder()
data[label_column] = label_encoder.fit_transform(data[label_column])

# Features and labels
X = data.drop([label_column], axis=1)
y = data[label_column]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE+Tomek
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'ddos_detection_model.pkl')
print("Model saved as 'ddos_detection_model.pkl'")

# Save the LabelEncoder for decoding predictions
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved as 'label_encoder.pkl'")
