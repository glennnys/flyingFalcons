import neuralNetwork
import numpy as np

inputSize = (200, 200)
outputSize = 40
pathToDataset = "C:\\Users\\glenn\\Documents\\dataset"
pathToStoreData = "C:\\Users\\glenn\\Documents"

# neuralNetwork.importFiles(pathToDataset, pathToStoreData, inputSize, outputSize)

print("loading X")
X = np.loadtxt(f'{pathToStoreData}\\converted dataset\\training set.csv', delimiter=',')
y = X[:, -1]
X = X[:, :-1]
print(X.shape[0], X.shape[1])

# X = np.ones([800, 400])
# y = np.ones([800, 1])

input_layer_size = X.shape[1]
hidden_layer_size = 100
hidden_layer_count = 2
num_labels = 40

newTheta = neuralNetwork.optimizeNN(200, input_layer_size, hidden_layer_size,
                                    hidden_layer_count,
                                    num_labels, X, y, 1, pathToStoreData)

correctness = neuralNetwork.validateFromFile(f'{pathToStoreData}\\optimized thetas\\trained thetas.csv',
                                             f'{pathToStoreData}\\converted dataset\\validating set.csv',
                                             input_layer_size, hidden_layer_size, hidden_layer_count, num_labels)
print(correctness)
