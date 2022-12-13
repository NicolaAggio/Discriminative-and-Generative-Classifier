from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# loads the dataset and return the train and test sets
def loadDataset():
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X/255

    X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)
    X_train, X_valid , y_train, y_valid = train_test_split(X_train_80, y_train_80, test_size = 0.50, random_state=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test