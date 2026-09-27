import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Dataset provided in the manual
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temp.': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
             'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, 
              False, False, False, True, True, False, True],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# 2. Encode categorical textual data into numerical labels
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Play', axis=1)
y = df['Play']

# 3. Train decision tree with random 10 data points and test with remaining 4
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10, test_size=4, random_state=42)

# 4. Create decision tree with different parameters and test it
parameters_list = [
    {"criterion": "gini", "max_depth": None},
    {"criterion": "entropy", "max_depth": None},
    {"criterion": "gini", "max_depth": 2},
    {"criterion": "entropy", "max_depth": 3}
]

# Print Results in the manual's requested format
print(f"{'Parameter':<35} | {'Accuracy of model'}")
print("-" * 55)

for params in parameters_list:
    # Initialize and train the model
    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Format the parameter string for output
    param_str = f"criterion={params['criterion']}, max_depth={params['max_depth']}"
    
    print(f"{param_str:<35} | {accuracy:.2f}")