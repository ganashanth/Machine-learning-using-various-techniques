from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

"""This code apply Decision Tree Classifier on the
given dataset. We have used train_test_split to split data into 
70/30 training and testing set. Hyperparameter optimization has 
been carried out using 10 fold cross validation on the training
 dataset. Running this file will print cross_validation 
 accuracy, test_accuracy and confusion matrix for test data."""

  #load the data and convert it in DataFrame object
data,meta = arff.loadarff("training_dataset.arff")
data = pd.DataFrame(data)

# Replace all negative values with '2'
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

#Decision Tree
#We will optimise criteria and max_depth value using 
#Hyper Parameter optimisation using 10 fold cross validation

max_depth_values = [3, 5, 10, 15]
criteria = ['gini', 'entropy']
cross_val_scores = []
for c in criteria:
        for d in max_depth_values:
                classifier = DecisionTreeClassifier(criterion = c, max_depth = d)
                crossval_score = cross_val_score(classifier, X_train.values, Y_train.values, cv = 10)
                cross_val_scores.append(crossval_score.mean())

max_score = max(cross_val_scores)
max_score_index = cross_val_scores.index(max_score)

print("using max_depth = 15 and criterion = 'gini' ")

classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 15)
classifier.fit(X_train.values, Y_train.values)
predicted = classifier.predict(X_test.values)

testing_accuracy = accuracy_score(Y_test.values, predicted)
conf_matrix = confusion_matrix(Y_test.values, predicted)

print("Cross Validation Score is %f"%max_score)
print("testing percentage accuracy is %f"%testing_accuracy)
print("confusionm matrix is")
print(conf_matrix)
