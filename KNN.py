from scipy.spatial import distance
from scipy.stats import mode
import numpy as np

# calculates the accuracy of the classifier
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))

# predicts the class of a new point
def predict(X_train, y, X_input, dist, k):
    predictions = []
     
    # loop through the datapoints to be classified
    for item in X_input: 
         
        # distances
        point_dist = []
         
        # loop through each training datapoint
        for j in range(len(X_train)): 
            # computation of the euclidean distance
            distances = distance.euclidean(np.array(X_train[j,:]) , item)
            point_dist.append(distances)
            
        point_dist = np.array(point_dist) 
         
        # sorting the array while preserving the index
        # keeping the first K datapoints
        dist = np.argsort(point_dist)[0:int(k)] 
         
        # labels of the K nearest datapoints 
        labels = y[dist]
         
        #  taking the majority voting
        lab = mode(labels) 
        lab = lab.mode[0]
        predictions.append(lab)
 
    return predictions