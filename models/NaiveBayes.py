import numpy as np
from statistics import mean, stdev
from scipy.stats import beta

# splits the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# calculates the mean, stdev and count for each feature of the dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# splits the dataset by class then calculate statistics for each row
def train_model(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# calculates the Beta probability distribution function for x
def calculate_probability(x, mean, stdev):
	epsilon = 0.05
	k = ((mean * (1-mean))/(stdev**2)) - 1
	alpha_ = k * mean
	beta_ = k * (1-mean)
	res = beta.cdf(x+epsilon, alpha_, beta_) - beta.cdf(x-epsilon, alpha_, beta_)

	res = np.nan_to_num(res, nan=1., copy=False)

	return res

# calculates the posterior probabilities for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		# prior probability
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, count = class_summaries[i]
			# class condition probability
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)

	return probabilities

# predicts the class of a given row by choosing the class with highest posterior probability
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
def NaiveBayes(train,test):
	summaries = train_model(train)
	predictions = []

	print('Starting the predictions..')

	for row in test:
		predictions.append(predict_class(summaries, row))

	return predictions

# calculates the accuracy of a classifier
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) 
