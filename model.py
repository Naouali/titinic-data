# Import libraries for mode
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd 


# Import  and preprocess the training and test data
Train = pd.read_csv("train.csv")
Test = pd.read_csv("test.csv")
X_train = Train.drop(['Survived'], axis=1)
#X_train = X_train.fillna("ffill")
X_train = X_train.drop(["Name"], axis=1)
Y_train = Train["Survived"]
Test = Test.drop(["Name"], axis=1)
print(X_train.columns)
print( Test.columns)
# Try to prepare one hot encoder for the data
X_train = pd.get_dummies(X_train)
Test = pd.get_dummies(Test)
print(X_train.shape, Test.shape)
print(X_train.head())