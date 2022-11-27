import neuralNetwork
import numpy as np

inputSize = (200, 200)
outputSize = 40
pathToDataset = "C:\\Users\\glenn\\OneDrive - KU Leuven\\Master Industrieel 1\\sem 1\\Machine Learning\\lab\\project\\dataset"
pathToStoreData = "C:\\Users\\glenn\\Documents"

# neuralNetwork.importFiles(pathToDataset, pathToStoreData, inputSize, outputSize)

X = np.loadtxt(f'{pathToStoreData}\\converted dataset\\training set.csv', delimiter=',')
y = X[:, -1]
X = X[:, :-1]

input_layer_size = X.shape[1]
hidden_layer_size = 100
hidden_layer_count = 2
num_labels = 40
thetas = neuralNetwork.randInitAllWeights(input_layer_size, hidden_layer_size, hidden_layer_count, num_labels)
initial_nn_params = np.concatenate([theta.ravel() for theta in thetas], axis=0)

newTheta = neuralNetwork.optimizeNN(200, input_layer_size, hidden_layer_size,
                                    hidden_layer_count,
                                    num_labels, X, y, 1, pathToStoreData)
