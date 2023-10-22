# Import necessary libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()

# Extract features (x) and target (y)
x = cancer.data[:, 3:4]  # Using one feature (index 3) for simplicity
y = (cancer.target == 0).astype(int)  # 0 for malignant, 1 for benign

# Train a Logistic Regression classifier
clf = LogisticRegression()
clf.fit(x, y)

# Define a new data point with the same number of features as in the training data
new_data_point = np.array([[1.6]])  # Example feature value for prediction

# Predict the class label for the new data point
predicted_class = clf.predict(new_data_point)

print("Predicted Class:", "Malignant" if predicted_class[0] == 0 else "Benign")

# Using matplotlib to visualize the probabilities
x_new = np.linspace(0, 5, 1000).reshape(-1, 1)  # Create 1000 data points in the range [0, 5]
y_prob = clf.predict_proba(x_new)  # Predict class probabilities

# Plotting the visualization
plt.plot(x_new, y_prob[:, 1], "g-", label="Benign (Class 0)")
plt.xlabel("Feature Value")
plt.ylabel("Probability of Malignant (Class 0)")
plt.legend()
plt.title("Logistic Regression Classifier")
plt.show()
