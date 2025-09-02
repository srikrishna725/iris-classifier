
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

parser = argparse.ArgumentParser(description="Train a Decision Tree on Iris dataset")
parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test set")
parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
args = parser.parse_args()

iris = load_iris()
X, y = iris.data, iris.target
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")



model = DecisionTreeClassifier(random_state=args.random_state)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

os.makedirs("../outputs", exist_ok=True)
plt.savefig("../outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to outputs/confusion_matrix.png")

print("Feature importances:", model.feature_importances_)
