# Step 1 - Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2 - Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Step 3 - Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 - Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5 - Make predictions
predictions = model.predict(X_test)

# Step 6 - Check accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7 - Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(new_flower)
print(f"Predicted flower type: {iris.target_names[result[0]]}")