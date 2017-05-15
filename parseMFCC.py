import numpy as np
from collections import Counter

# XTrain = np.load('./Data/X_Train.npy')
# YTrain = np.load('./Data/Y_Train.npy')

# XTest = np.load('./Data/X_Test.npy')
# YTest = np.load('./Data/Y_Test.npy')

# XTrain = np.load('./Data/X_Train_Middle.npy')
# YTrain = np.load('./Data/Y_Train_Middle.npy')

# XTest = np.load('./Data/X_Test_Middle.npy')
# YTest = np.load('./Data/Y_Test_Middle.npy')

# XTrain = np.load('./Data/X_Train_WithCov.npy')
# YTrain = np.load('./Data/Y_Train_WithCov.npy')

# XTest = np.load('./Data/X_Test_WithCov.npy')
# YTest = np.load('./Data/Y_Test_WithCov.npy')

# XTrain = np.load('./Data/X_Train_WithReggae.npy')
# YTrain = np.load('./Data/Y_Train_WithReggae.npy')

# XTest = np.load('./Data/X_Test_WithReggae.npy')
# YTest = np.load('./Data/Y_Test_WithReggae.npy')

# X = np.load('./Data/X.npy')
X_first = np.load('./Data/X_first.npy')
X_middle = np.load('./Data/X_middle.npy')
X_end = np.load('./Data/X_end.npy')
Y = np.load('./Data/Y.npy')

def getDifference(array1, array2, numFeatures):
	return np.subtract(array1[0:numFeatures], array2[0:numFeatures])

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
		# if genre != "Jazz":
		tuples.append((key, genre))
	return tuples

def getClosestArrays(array, otherArrays, k, numFeatures):
	tuples = []
	for a in otherArrays:
		if not np.array_equal(array, a[0]):
			diff = getDifference(array, a[0], numFeatures)
			square = squareArray(diff, 2)
			squareSum = sumOfArray(square)
			tuples.append((a[0], a[1], squareSum))
	sortedTuples = sorted(tuples, key = lambda x: x[2])
	return sortedTuples[:k]

def getModeOfTuples(tuples):
	groups = [t[1] for t in tuples]
	data = Counter(groups)
	return data.most_common(3)

def printDataInfo(data):
	for key in data.keys():
		value = data[key]
		percentDict = {}
		total = sum(value.values())
		for k in value.keys():
			v = value[k]
			percentDict[k] = 100.0 * v / float(total)
		print key
		print str(data[key])
		print str(percentDict)
		print

def getCorrect(xTrain, yTrain, xTest, yTest, k, numFeatures):
	data = {}
	trainArray = makeArrayTuples(xTrain, yTrain)
	testArray = makeArrayTuples(xTest, yTest)
	correctCount = 0
	for a in testArray:
		actual = a[1]
		closestArrays = getClosestArrays(a[0], trainArray, k, numFeatures)
		mode = getModeOfTuples(closestArrays)
		expected = mode[0][0]
		if expected == actual:
			correctCount += 1
		dataValue = data.get(actual, {})
		dataValue[expected] = dataValue.get(expected, 0) + 1
		data[actual] = dataValue
	return 100 * correctCount/float(len(testArray))
	# printDataInfo(data)
	# print 100 * correctCount/float(len(testArray))

def getCorrect_Full(x_first, x_middle, x_end, y, k, numFeatures):
	data = {}
	firstArray = makeArrayTuples(x_first, y)
	middleArray = makeArrayTuples(x_middle, y)
	endArray = makeArrayTuples(x_end, y)
	correctCount = 0
	for i in range(len(firstArray)):
		first = firstArray[i]
		middle = middleArray[i]
		end = endArray[i]
		actual = first[1]
		closestArrays = getClosestArrays(first[0], firstArray, k, numFeatures) + getClosestArrays(middle[0], middleArray, k, numFeatures) + getClosestArrays(end[0], endArray, k, numFeatures)
		mode = getModeOfTuples(closestArrays)
		expected = mode[0][0]
		if expected == actual:
			correctCount += 1
		dataValue = data.get(actual, {})
		dataValue[expected] = dataValue.get(expected, 0) + 1
		data[actual] = dataValue
	return 100 * correctCount/float(len(firstArray))
	# printDataInfo(data)
	# print 100 * correctCount/float(len(testArray))

# Start
# k = 14, numFeatures = 7, 47.5
# Without Jazz
# k = 10, numberFeatures = 12

# Middle
# k = 19, numberFeatures = 3
# Without Jazz
# k = 17, numberFeatures = 5

# WithCov
# k = 7, numberFeatures = 68
# Without Jazz
# k = 3, numberFeatures = 95

# WithReggae
# k = 3, numberFeatures = 96

# Knife Method
# k = 8, numberFeatures = 34

# getCorrect(XTrain, YTrain, XTest, YTest, 3, 96)
# getCorrect(X, Y, X, Y, 8, 34)

bestValue = 0
bestIndex = (None, None)

for i in range(1, 10):
	print "K = " + str(i)
	for j in range(3, 98):
		value = getCorrect(X_first, X_middle, X_end, Y, i, j)
		if value > bestValue:
			bestValue = value
			print bestValue
			bestIndex = (i, j)

print "Best Accuracy: " + str(bestValue)
print "Best Index: " + str(bestIndex)




