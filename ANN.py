from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


"""This code apply MLPClassifier(Multi Layer Perceptron)
 Algorithm on the given dataset. We have used train_test_split
to split data into 70/30 training and testing set. Hyperparameter 
 optimization has been carried out using 10 fold cross validation on 
 the training dataset. Running this file will print cross_validation 
 accuracy, test_accuracy and confusion matrix for test data."""

 #load the data and convert it in DataFrame object
data,meta = arff.loadarff("training_dataset.arff")
data = pd.DataFrame(data)

#We need to replace all negative values with '2'
data = data.replace('-1', '2')
data = pd.get_dummies(data, columns = ['URL_Length','SSLfinal_State','having_Sub_Domain',\
'URL_of_Anchor','Links_in_tags','SFH', 'web_traffic',\
'Links_pointing_to_page'])
data = data.apply(pd.to_numeric)

#Creating predictors and target
labels = data.columns
X = data[labels[:-1]]
Y = data['Result']

#splitting into train/test set (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

#Aritificial Neural Net
# We have two hyper parameter in MLPClassifier
#hidden layer size and learning rate
# we will optimise these using hyperparameter optimization

hidden_layer_values = [10, 25, 50, 100]
learning_rate_values = [0.0001, 0.001, 0.01, 0.1, 1]
cross_val_scores = []
for h in hidden_layer_values:
        for l in learning_rate_values:
                classifier = MLPClassifier(hidden_layer_sizes = h, learning_rate_init = l)
                crossval_score = cross_val_score(classifier, X_train.values, Y_train.values, cv = 10)
                cross_val_scores.append(crossval_score.mean())

max_score = max(cross_val_scores)
max_score_index = cross_val_scores.index(max_score)

print("using hidden_layer_size = 50 and learning_rate = 0.1")

classifier = MLPClassifier(hidden_layer_sizes = 50, learning_rate_init = 0.1)
classifier.fit(X_train.values, Y_train.values)
predicted = classifier.predict(X_test.values)

testing_accuracy = accuracy_score(Y_test.values, predicted)
conf_matrix = confusion_matrix(Y_test.values, predicted)

print("Cross Validation Score is %f"%max_score)
print("testing percentage accuracy is %f"%testing_accuracy)
print("confusionm matrix is")
print(conf_matrix)
