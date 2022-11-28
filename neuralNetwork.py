import numpy as np
from scipy import optimize
import csv
import os
from PIL import Image, ImageFile
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============neural network functions==================
def validateFromValues(optimizedThetas, X, y):
    print("validating")
    p = predict(optimizedThetas, X)
    correctCount = 0
    for i in range(len(p)):
        if p[i] == y[i]:
            correctCount += 1

    return correctCount / len(p)


def validateFromFile(pathToThetas, pathToX, input_layer_size, hidden_layer_size, hidden_layer_count, num_labels):
    print("loading X values")
    X = np.loadtxt(pathToX, delimiter=',')
    print("loading thetas")
    optimizedThetas = nnparamsToThetas(np.loadtxt(pathToThetas, delimiter=','), input_layer_size, hidden_layer_size,
                                       hidden_layer_count, num_labels)
    print(optimizedThetas[0].shape)
    y = X[:, -1]
    X = X[:, :-1]

    return validateFromValues(optimizedThetas, X, y)


# might work
def predict(thetas, X):
    # Make sure the input has two dimensions
    print("predicting output")
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions

    # useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(X.shape[0])
    for i in range(m):
        a1 = X[i]
        a1 = np.insert(a1, 0, 1)
        aPrev = a1
        # TODO test if this part works as intended
        print(len(thetas))
        for j in range(len(thetas)):
            aNext = sigmoid(np.dot(thetas[j], aPrev))
            aNext = np.insert(aNext, 0, 1)
            aPrev = aNext
        p[i] = np.argmax(aPrev)
    return p


# needs to be written in a way that it takes any amount of thetas (btw, all hidden layers are the same size in our case)
def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size, hidden_layer_count,
                   num_labels,
                   X, y, lambda_=0.0):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network

    thetas = nnparamsToThetas(nn_params, input_layer_size, hidden_layer_size, hidden_layer_count, num_labels)

    # Setup some useful variables
    m = y.size

    # You need to return the following variables correctly
    J = 0

    temp = np.zeros([y.shape[0], num_labels])
    for i in range(m):
        temp[i, int(y[i] - 1)] = 1

    a = []
    z = []
    a.append(np.transpose(X))
    a[0] = np.concatenate([np.ones((1, a[0].shape[1])), a[0]], axis=0)
    for i in range(len(thetas)):
        z.append(np.dot(thetas[i], a[i]))
        a.append(sigmoid(z[i]))
        a[i + 1] = np.concatenate([np.ones((1, a[i + 1].shape[1])), a[i + 1]], axis=0)

    h = np.transpose(sigmoid(z[-1]))

    regTerm = 0
    for theta in thetas:
        regTerm += (lambda_ / (2 * m)) * np.sum(theta[:, 1:] ** 2)

    J = (np.sum(-temp * np.log(h) - ((1 - temp) * np.log(1 - h)))) / m + regTerm

    d = []
    thetaGrads = []
    d.append(np.transpose(h - temp))
    for i in range(len(thetas) - 1, 0, -1):
        d.insert(0, np.dot(np.transpose(thetas[i][:, 1:]), d[0]) * sigmoidGradient(z[i - 1]))
    for i in range(len(thetas)):
        thetaGrads.append(np.dot(d[i], np.transpose(a[i])) / m)

    for i in range(len(thetas)):
        thetaGrads[i] += (lambda_ / m) * np.concatenate([np.zeros((thetas[i][:, 1:].shape[0], 1)), thetas[i][:, 1:]],
                                                        axis=1)

    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([thetaGrad.ravel() for thetaGrad in thetaGrads])
    return J, grad


# I think this works
def optimizeNN(maxIter, input_layer_size, hidden_layer_size,
               hidden_layer_count,
               num_labels, X, y, lambda_, pathToStoreThetas):
    print("optimizing thetas")
    options = {'maxiter': maxIter}
    if not os.path.isdir(f"{pathToStoreThetas}\\optimized thetas"):
        os.mkdir(f"{pathToStoreThetas}\\optimized thetas")

    costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                            hidden_layer_size,
                                            hidden_layer_count,
                                            num_labels, X, y, lambda_)

    start_time = time.time()
    initialThetas = randInitAllWeights(input_layer_size, hidden_layer_size, hidden_layer_count, num_labels)
    initial_nn_params = np.concatenate([theta.ravel() for theta in initialThetas], axis=0)
    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)
    print(f"it took {time.time() - start_time} seconds to optimize neural network")
    nn_params = res.x
    thetas = nnparamsToThetas(nn_params,
                              input_layer_size,
                              hidden_layer_size, hidden_layer_count,
                              num_labels)

    np.savetxt(f'{pathToStoreThetas}\\optimized thetas\\trained thetas.csv', nn_params, delimiter=',')
    print("thetas optimized")
    return thetas


def importFiles(pathToDataset, pathToStoreData, input_size, output_size):
    path = pathToDataset
    inputSize = input_size
    outputSize = output_size
    start_time = time.time()
    os.chdir(path)
    if not os.path.isdir(f"{pathToStoreData}\\converted dataset"):
        os.mkdir(f"{pathToStoreData}\\converted dataset")
    # iterate through all file
    i = 0
    filename, width, height, planeType, xmin, ymin, xmax, ymax = 0, 0, 0, 0, 0, 0, 0, 0
    fileCount = len(os.listdir())

    planeDict = {}
    X = []
    y = []
    training = 1
    validating = 0
    testing = 0
    for file in os.listdir():
        # Check whether file is in text format or not
        rows = []
        if file.endswith(".csv"):
            csvFile = open(file)
            csvreader = csv.reader(csvFile)
            header = next(csvreader)

            for row in csvreader:
                rows.append(row)
            filename, width, height, planeType, xmin, ymin, xmax, ymax = rows[0]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            if planeType not in planeDict:
                planeDict[planeType] = len(planeDict) + 1
            y = planeDict[planeType]

        if file.endswith(".jpg"):
            img = Image.open(file)
            imageFileName = file.replace('.jpg', '')
            if imageFileName == filename:
                cropImg = img.crop((xmin, ymin, xmax, ymax))
                # resize could be improved to keep aspect ratio using thumbnail function for example
                resizedImg = cropImg.resize(inputSize)
                recoloured = resizedImg.convert('1')
                numpydata = np.asarray(recoloured)
                if testing != 1:
                    numpydata = np.append(numpydata, y)
                X.append(numpydata.flatten())
                # make csv files from dataset
                if i >= fileCount * 0.6 and training == 1:
                    training = 0
                    validating = 1
                    np.savetxt(f'{pathToStoreData}\\converted dataset\\training set.csv', X, delimiter=',')
                    X = []
                    y = []
                    print("created training set")
                if i >= fileCount * 0.8 and validating == 1:
                    validating = 0
                    testing = 1
                    np.savetxt(f'{pathToStoreData}\\converted dataset\\validating set.csv', X, delimiter=',')
                    X = []
                    y = []
                    print("created validating set")
        i += 1

    np.savetxt(f'{pathToStoreData}\\converted dataset\\test set.csv', X, delimiter=',')
    print("created test set")
    print(f"it took {time.time() - start_time} seconds to import {fileCount} data samples")


# ============basic functions==================
# definitely works(overflows?)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# definitely works
def sigmoidGradient(z):
    gz = sigmoid(z)
    g = gz * (1 - gz)
    return g


# definitely works
def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init


# this works
def randInitAllWeights(input_layer_size, hidden_layer_size,
                       hidden_layer_count,
                       num_labels):
    thetas = []
    for layer in range(hidden_layer_count + 1):
        if layer == 0:
            newTheta = randInitializeWeights(input_layer_size, hidden_layer_size)
        elif layer == hidden_layer_count:
            newTheta = randInitializeWeights(hidden_layer_size, num_labels)
        else:
            newTheta = randInitializeWeights(hidden_layer_size, hidden_layer_size)
        thetas.append(newTheta)

    return thetas


# this works
def nnparamsToThetas(nn_params,
                     input_layer_size,
                     hidden_layer_size, hidden_layer_count,
                     num_labels):
    thetas = []
    for layer in range(hidden_layer_count + 1):
        if layer == 0:
            thetas.append(np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                                     (hidden_layer_size,
                                      (input_layer_size + 1))))

        elif layer == hidden_layer_count:
            thetas.append(np.reshape(nn_params[((hidden_layer_size
                                                 * (input_layer_size + 1))
                                                + ((hidden_layer_size ** 2
                                                    + hidden_layer_size) * (hidden_layer_count - 1))):],
                                     (num_labels, (hidden_layer_size + 1))))

        else:
            thetas.append(np.reshape(nn_params[((hidden_layer_size * (input_layer_size + 1))
                                                + ((hidden_layer_size ** 2 + hidden_layer_size) * (layer - 1))):
                                               ((hidden_layer_size * (input_layer_size + 1))
                                                + ((hidden_layer_size ** 2 + hidden_layer_size) * layer))],
                                     (hidden_layer_size, (hidden_layer_size + 1))))
    return thetas
