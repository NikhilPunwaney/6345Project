import numpy as np
from collections import Counter

XTrain = np.load('./Data/X_Train.npy')
YTrain = np.load('./Data/Y_Train.npy')

XTest = np.load('./Data/X_Test.npy')
YTest = np.load('./Data/Y_Test.npy')

def getDifference(array1, array2):
	return np.subtract(array1[0:5], array2[0:5])

def squareArray(array, power):
	return np.power(array, power)

def sumOfArray(array):
	return np.sum(array)

def arrayGenre(array):
	if array[0] == 1:
		return "Classical"
	if array[1] == 1:
		return "Jazz"
	if array[2] == 1:
		return "Metal"
	if array[3] == 1:
		return "Pop"

def makeArrayTuples(keys, labels):
	tuples = []
	for k in range(len(keys)):
		key = keys[k]
		label = labels[k]
		genre = arrayGenre(label)
		tuples.append((key, genre))
	return tuples

def getClosestArrays(array, otherArrays, k):
	tuples = []
	for a in otherArrays:
		diff = getDifference(array, a[0])
		square = squareArray(diff, 2)
		squareSum = sumOfArray(square)
		tuples.append((a[0], a[1], squareSum))
	sortedTuples = sorted(tuples, key = lambda x: x[2])
	return sortedTuples[:k]

def getModeOfTuples(tuples):
	groups = [t[1] for t in tuples]
	data = Counter(groups)
	return data.most_common(3)

def getCorrect(xTrain, yTrain, xTest, yTest, k):
	trainArray = makeArrayTuples(xTrain, yTrain)
	testArray = makeArrayTuples(xTest, yTest)
	correctCount = 0
	for a in testArray:
		closestArrays = getClosestArrays(a[0], trainArray, k)
		mode = getModeOfTuples(closestArrays)
		if mode[0][0] == a[1]:
			correctCount += 1
	return correctCount/float(len(yTest))

for i in range(1, 20):
	print getCorrect(XTrain, YTrain, XTest, YTest, i)





