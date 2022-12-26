from scipy.spatial import distance
import numpy as np

# returns the k-nearest data points of the given point according to the provided metric. The default metric is 'euclidean'
def get_neighbors(point, X, k, metric):
    distances = []
    neighbors = []
    
    for i in range(len(X)):
        elem = X[i]
        
        if metric == 'euclidean':
                dist = distance.euclidean(point, elem)
        else:
            if metric == 'cosine':
                dist = distance.cosine(point, elem)
            else:
                if metric == 'manhattan':
                    dist = distance.cityblock(point, elem)
                else:
                    dist = distance.euclidean(point, elem)
        distances.append((elem, dist, i))
        distances.sort(key = lambda tupl : tupl[1])
    
    for i in range(k + 1):
        neighbors.append((distances[i][0], distances[i][2]))
        
    return neighbors

# classifies the given point according to majority vote of the k-nearest neighobors
def predict(point, X, y, k, metric):
    classes = []
    neighbors = get_neighbors(point, X, k, metric)
    
    for neighbour in neighbors:
        pos = neighbour[1]
        classes.append(y[pos])
        
    return max(set(classes), key = classes.count)

# k-nearest neighbors algorithm implementation
def KNN(X, y, k, metric):
    classifications = []
    for i in range (X.shape[0]):
        point = X[i]
        classifications.append(predict(point, X, y, k, metric))
        
    return classifications 

# calculates the accuracy of a classifier
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) 