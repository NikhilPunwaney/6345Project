import numpy as np

classicalTraining = np.load('./Data/classical_training.npy')
classicalTraining = np.load('./Data/classical_training.npy')
classicalTraining = np.load('./Data/classical_training.npy')
classicalTraining = np.load('./Data/classical_training.npy')

classicalTest = np.load('./Data/classical_test.npy')
classicalTest = np.load('./Data/classical_test.npy')
classicalTest = np.load('./Data/classical_test.npy')
classicalTest = np.load('./Data/classical_test.npy')


def getDifference(array1, array2):
	return np.subtract(array1[0], array2[1])

def squareArray(array, power):
	return np.power(array, power)

def sumOfArray(array):
	return np.sum(array)

def getClosestArray(array, listOfArrays):
	minArray = None
	minDifference = float("inf")
	for a in listOfArrays:
		diff = getDifference(array, a)
		square = squareArray(diff, 2)
		squareSum = sumOfArray(square)
		if squareSum < minDifference:
			minArray = a
			minDifference = squareSum
	return (minArray, minDifference)

def getGenre(array, listOfGenres):
	minArray = None
	minDifference = float("inf")
	closestGenre = None
	for index in range(len(listOfGenres)):
		genre = listOfGenres[index]
		closestArray = getClosestArray(array, genre)
		if closestArray[1] < minDifference:
			minArray = closestArray[0]
			minDifference = closestArray[1]
			closestArray = index
	return (closestArray, minArray, minDifference)
