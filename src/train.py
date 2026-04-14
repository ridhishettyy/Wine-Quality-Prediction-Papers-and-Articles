import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from preprocessing import load_data, create_health_label

# Load data
df = load_data("data/winequality.csv")
df = create_health_label(df)
#this has been added temporarily
print(df['health_risk'].value_counts())
# Features & target
X = df.drop(['health_risk', 'quality'], axis=1)
y = df['health_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for SVM & KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SVM": SVC(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0

print("Model Performance:\n")

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save best model
joblib.dump(best_model, "models/model.pkl")

print(f"\nBest model saved with accuracy: {best_accuracy:.4f}")