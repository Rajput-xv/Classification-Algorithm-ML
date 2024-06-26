import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


#loding & preparing data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#implementation & training model

#decision tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
#Naive bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
#K- Nereast Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
predictions_dtree = dtree.predict(X_test)
predictions_nb = nb.predict(X_test)
predictions_knn = knn.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, predictions_dtree))
print("Naïve Bayes Accuracy:", accuracy_score(y_test, predictions_nb))
print("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, predictions_knn))

# Confusion Matrix and Classification Report
print("\nConfusion Matrix - Decision Tree")
print(confusion_matrix(y_test, predictions_dtree))
print(classification_report(y_test, predictions_dtree))

print("\nConfusion Matrix - Naïve Bayes")
print(confusion_matrix(y_test, predictions_nb))
print(classification_report(y_test, predictions_nb))

print("\nConfusion Matrix - K-Nearest Neighbors")
print(confusion_matrix(y_test, predictions_knn))
print(classification_report(y_test, predictions_knn))



