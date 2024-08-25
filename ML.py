import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Importing csv file
h = pd.read_csv("data.csv")

# Null Values
h.isnull()

# Drop
X = h.drop("label", axis=1)
y = h["label"]

# Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prediction
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Amount values
N = float(input("Enter the Nitrogen value: "))
P = float(input("Enter the Phosphorus value: "))
K = float(input("Enter the Potassium value: "))
temperature = float(input("Enter the temperature value: "))
humidity = float(input("Enter the humidity value: "))
ph = float(input("Enter the pH value: "))
rainfall = float(input("Enter the rainfall value: "))
user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                          columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
predicted_crop_label = tree_clf.predict(user_input)
print("Predicted Crop Label:", predicted_crop_label[0])
