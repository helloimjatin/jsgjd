# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset (update path if needed)
df = pd.read_csv(r"C:\Users\patha\Downloads\symptom_disease_dataset_500.csv")  # Ensure CSV is in the same folder

# Step 2: Split features and target
X = df.drop("disease", axis=1)
y = df["disease"]

# Step 3: Encode disease labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Train Decision Tree model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📝 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save model and encoder
joblib.dump(clf, "disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Model and label encoder saved successfully!")
