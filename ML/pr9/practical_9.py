from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Divide dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Neural networks are highly sensitive to unscaled data, so we standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define different architectures (hidden layers) and training functions (solvers/optimizers)
experiments = [
    {"hidden_layer_sizes": (10,), "solver": "adam"},
    {"hidden_layer_sizes": (10,), "solver": "sgd"},
    {"hidden_layer_sizes": (10, 10), "solver": "adam"},
    {"hidden_layer_sizes": (10, 10), "solver": "lbfgs"},
    {"hidden_layer_sizes": (50,), "solver": "adam"}
]

# Print Results in the manual's requested format
print(f"{'Architecture of Neural network':<35} | {'Training function':<20} | {'Accuracy'}")
print("-" * 70)

for exp in experiments:
    # Initialize and train the neural network model
    # max_iter is set high to ensure the network converges during training
    mlp = MLPClassifier(
        hidden_layer_sizes=exp["hidden_layer_sizes"], 
        solver=exp["solver"], 
        max_iter=2000, 
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    # Predict and calculate accuracy on the 30% test data
    y_pred = mlp.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Format architecture tuple as string (e.g., "(10,) nodes")
    arch_str = f"{exp['hidden_layer_sizes']} nodes"
    
    print(f"{arch_str:<35} | {exp['solver']:<20} | {accuracy:.4f}")