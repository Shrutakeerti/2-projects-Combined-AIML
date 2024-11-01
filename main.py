# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and preprocess the dataset
data = pd.read_csv(r'D:\Jal Shakti\District_Statewise_Well.csv')

# Create a binary target column based on 'Stage of Ground Water Extraction (%)'
data['High_Extraction'] = (data['Stage of Ground Water Extraction (%)'] > 50).astype(int)

# Drop irrelevant columns
data_cleaned = data.drop(columns=['S.no.', 'Name of State', 'Name of District', 'Stage of Ground Water Extraction (%)'])

# Separate features and target variable
X = data_cleaned.drop(columns=['High_Extraction'])
y = data_cleaned['High_Extraction']

# Fill missing values with the mean for each column
X = X.fillna(X.mean())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
model_path = 'well_extraction_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
