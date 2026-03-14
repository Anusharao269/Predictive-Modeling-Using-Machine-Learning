import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=10)

# Train models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# Accuracy
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

# Accuracy graph
models = ["Decision Tree", "Random Forest"]
accuracies = [dt_acc, rf_acc]

plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()