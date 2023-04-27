# CodeClause_WineQuality-Prediction
wine quality Prediction program that uses a Random Forest Classifier algorithm to train a model on a wine quality dataset. 
Description-
 The wine quality data is stored in a CSV file named 'wine_quality.csv'. The first step is to load the data into a pandas DataFrame. We then split the dataset into training and testing sets using the train_test_split function from scikit-learn.

Next, we create an instance of the RandomForestClassifier class and fit it to the training data using the fit method. We then use the trained model to make predictions on the test set using the predict method.

Finally, we evaluate the accuracy of the model using the accuracy_score function from scikit-learn, which compares the predicted labels to the true labels and returns the fraction of correct predictions. The accuracy is printed to the console.
