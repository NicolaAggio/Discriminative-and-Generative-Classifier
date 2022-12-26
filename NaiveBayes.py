import numpy as np
from statistics import mean, stdev
from scipy.stats import beta


# Splits the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculates the mean, stdev and count for each column in the dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Splits the dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculates the Beta probability distribution function for x
def calculate_probability(x, mean, stdev):
	k = ((mean * (1-mean))/(stdev**2)) - 1
	alpha_ = k * mean
	beta_ = k * (1-mean)
	res = beta.pdf(x, alpha_, beta_, loc = 0, scale = 1)
	print(res)
	return res

# Calculates the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, count = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    
	return probabilities

# Predicts the class of a given row
def predict_class(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_class = None
    best_prob = -1

    for class_,prob in probabilities.items():
        if best_class == None or prob > best_prob:
            best_class = class_
            best_prob = prob
    
    return best_class

# NaiveBayes classifier
def NaiveBayes(test):
    summaries = summarize_by_class(test)
    predictions = []

    for row in test:
        predictions.append(predict_class(summaries, row))

    return predictions

# Calculates the accuracy of a classifier
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) 

dataset = np.array([[3.393533211,2.331273381,0],
 [3.110073483,1.781539638,0],
 [1.343808831,3.368360954,0],
 [3.582294042,4.67917911,0],
 [2.280362439,2.866990263,0],
 [7.423436942,4.696522875,1],
 [5.745051997,3.533989803,1],
 [9.172168622,2.511101045,1],
 [7.792783481,3.424088941,1],
 [7.939820817,0.791637231,1]])

# print(dataset.shape)
# print(list(dataset[:,2]))
# predictions = NaiveBayes(dataset)
# print('precitions = ',predictions)
# print('accuracy: ',accuracy(dataset[:,2],predictions) * 100)

# print(calculate_class_probabilities(summarize_dataset(dataset),dataset))

print(calculate_class_probabilities(summarize_by_class(dataset), dataset[0]))