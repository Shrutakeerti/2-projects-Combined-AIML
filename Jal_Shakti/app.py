import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


data = pd.read_csv(r'D:\Jal Shakti\District_Statewise_Well.csv')


data['High_Extraction'] = (data['Stage of Ground Water Extraction (%)'] > 50).astype(int)


data_cleaned = data.drop(columns=['S.no.', 'Name of State', 'Name of District', 'Stage of Ground Water Extraction (%)'])


X = data_cleaned.drop(columns=['High_Extraction'])
y = data_cleaned['High_Extraction']


X = X.fillna(X.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


model_path = 'well_extraction_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")
