# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:22:13 2018

@author: Aravind
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

# y is of int type. Change it to categorical
y = y.astype('object')

# Visualization
from matplotlib import pyplot as plt
m = X.shape[0]
n = X.shape[1]
labels = np.unique(y)
labels_count = labels.shape[0]

    
# Creating and plotting average digits
average_digits = np.empty((0, n+1))

plt.figure(figsize=(8,7))
plt.gray()

for label in labels:
    digits = X[y.flatten() == label]
    average_digit = digits.mean(0)   
    average_digits = np.vstack((average_digits, np.append(average_digit, label)))
    image = average_digit.reshape(28, 28)
    plt.subplot(3,4,label+1)
    plt.imshow(image)
    plt.title('Average '+str(label))
plt.show()

average_digits_x = average_digits[:,:-1]
average_digits_y = average_digits[:,-1]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#  one hot encoding
from keras.utils import np_utils
y = np_utils.to_categorical(y)

# Splitting the dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import Keras libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense

# ANN
classifier = Sequential()

classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 50, epochs = 10) # Lesser no of epochs - Basic Model

# Prediction
y_pred = classifier.predict(X_test)

maxi = y_pred.max(axis=1)
for i in range(len(y_pred)):
    for j in range(10):
        if y_pred[i,j] == maxi[i]:
           y_pred[i,j] = 1
        else:
               y_pred[i,j] = 0
     
# Accuracy    
crt_values = (y_pred == y_test).sum()
wrong_values = (y_pred != y_test).sum()
total = crt_values+wrong_values
result = crt_values/total
print(result) # 99.3% accuracy


# submission
test = pd.read_csv("test.csv")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.transform(test)

# fit ann
classifier.fit(X, y, batch_size = 50, epochs = 25)

# Prediction
y_pred_test = classifier.predict(test)

test_labels = np.argmax(y_pred_test, axis=1)

test_labels = pd.Series(test_labels, name="Label")

final_submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), test_labels], axis = 1)

# Submission file

final_submission.to_csv("Digit_Recognition_ANN_3.csv", index=False)#0.97

