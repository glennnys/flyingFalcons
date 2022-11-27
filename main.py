import os

import neuralNetwork
import numpy as np
import csv

def readFile(path):
    os.chdir(path)
    rows = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csvFile = open(file)
            csvreader = csv.reader(csvFile)
            header = next(csvreader)
    for row in csvreader:
        if len(row) > 0:
            X,y = row
            print(len(X), y)
        rows.append(row)

inputSize = (50, 50)
outputSize = 40
cwd = os.getcwd()
neuralNetwork.importFiles(cwd, "dataset", inputSize, outputSize)
#readFile(cwd)

# testing neural network
X = np.ones((23, 400))
y = np.asarray([2, 3, 1, 3, 5, 7, 9, 6, 3, 2, 5, 9, 0, 8, 6, 3, 5, 8, 4, 4, 6, 7, 8])

input_layer_size = X.shape[1]
hidden_layer_size = 25
hidden_layer_count = 10
num_labels = 10
thetas = neuralNetwork.randInitAllWeights(input_layer_size, hidden_layer_size, hidden_layer_count, num_labels)
initial_nn_params = np.concatenate([theta.ravel() for theta in thetas], axis=0)

initial_cost = neuralNetwork.nnCostFunction(initial_nn_params,
                                            input_layer_size,
                                            hidden_layer_size, hidden_layer_count,
                                            num_labels,
                                            X, y)

newTheta = neuralNetwork.optimizeNN(200, input_layer_size, hidden_layer_size,
                                    hidden_layer_count,
                                    num_labels, X, y, 1)
