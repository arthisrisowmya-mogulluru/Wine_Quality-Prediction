# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading data from CSV file
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

# Checking the data
print(wine_data.head())

# Checking the shape of the data
print(wine_data.shape)

# Visualizing the distribution of wine quality
plt.hist(wine_data['quality'], bins=6, color='blue')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality')
plt.show()

# Splitting the data into training and testing sets
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Decision Tree model
wine_model = DecisionTreeClassifier(max_depth=3, random_state=42)
wine_model.fit(X_train, y_train)

# Predicting the quality of wine
y_pred = wine_model.predict(X_test)

# Checking the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model:', accuracy)

# Creating a confusion matrix to visualize the performance of the model
confusion = confusion_matrix(y_test, y_pred)
plt.matshow(confusion)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

