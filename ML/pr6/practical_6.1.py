# Implementation: Logistic Regression Model and Performance Metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Iris flower data set
iris = load_iris()
X = iris.data
y = iris.target

# 2. Divide dataset in 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 3. Use 70% data to train logistic regression model
# Note: max_iter is set to 200 to ensure convergence without warnings
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Use 30% data to test the model performance
y_pred = model.predict(X_test)

# 5. Measure various performance metrics
# Since Iris has 3 classes, we use average='weighted' for precision, recall, and F1
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print results in a formatted table
print(f"{'Metric':<15} | {'Value'}")
print("-" * 30)
print(f"{'Precision':<15} | {precision:.4f}")
print(f"{'Recall':<15} | {recall:.4f}")
print(f"{'F1 Score':<15} | {f1:.4f}")
print(f"{'Accuracy':<15} | {accuracy:.4f}")