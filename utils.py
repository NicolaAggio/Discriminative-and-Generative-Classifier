from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# loads the dataset and return the train and test sets
def loadDataset():
    # importing the dataset
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    
    # deleting the columns with unique values and rescaling
    X = X[:, ~np.all(X[1:] == X[:-1], axis=0)]
    X = X/255
    
    # splitting the data into train, validation and test
    X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)
    X_train, X_valid , y_train, y_valid = train_test_split(X_train_80, y_train_80, test_size = 0.50, random_state=0)
    
    X.to_csv(r'./dataset/X.csv',index=False)
    y.to_csv(r'./dataset/y.csv',index=False)
    
    X_train.to_csv(r'./dataset/X_train.csv',index=False)
    X_valid.to_csv(r'./dataset/X_valid.csv',index=False)
    X_test.to_csv(r'./dataset/X_test.csv',index=False)
    
    y_train.to_csv(r'./dataset/y_train.csv',index=False)
    y_valid.to_csv(r'./dataset/y_valid.csv',index=False)
    y_test.to_csv(r'./dataset/y_test.csv',index=False)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
