import time
import numpy as np
from PIL import ImageFilter, Image
from scipy import optimize
import cv2 as cv2


# ============neural network functions==================
# probably works
def predict(thetas, X):
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions

    # useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(X.shape[0])
    for i in range(m):
        a = [X[i]]
        for j in range(len(thetas)):
            a[j] = np.insert(a[j], 0, 1)
            a.append(sigmoid(np.dot(thetas[j], a[j])))
        p[i] = np.argmax(a[-1])
    return p


def predictFromCamera(thetas, input_size):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        processedImg = Image.fromarray(img).resize(input_size).convert('L').filter(ImageFilter.FIND_EDGES)
        file_data = np.asarray(processedImg)
        file_data = file_data.flatten()
        print(f"{predict(thetas, file_data)}", end='\r', flush=True)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def findGoodValues(input_layer_size, num_labels, XTrain, yTrain, XVal, yVal):
    maxiter = 100
    bestTheta = []
    bestSize = 0
    bestCount = 0
    bestPrediction = 0

    print("estimated time: Very long")
    startTime = time.time()
    for hidden_layer_size in range(100, 2100, 100):
        for hidden_layer_count in range(1, 2):
            for lambda_ in range(0, 6):
                print(f'lambda: {lambda_}, layer count: {hidden_layer_count}, layer size: {hidden_layer_size}')
                oldTheta = optimizeNN(maxiter, input_layer_size, hidden_layer_size, hidden_layer_count, num_labels,
                                      XTrain, yTrain, lambda_, False, 0)
                shapedTheta = nnparamsToThetas(oldTheta, input_layer_size, hidden_layer_size, hidden_layer_count,
                                               num_labels)

                oldIterPred = np.mean(predict(shapedTheta, XVal) == yVal) * 100
                while True:
                    newTheta = optimizeNN(maxiter, input_layer_size, hidden_layer_size, hidden_layer_count, num_labels,
                                          XTrain, yTrain, lambda_, True, oldTheta)
                    shapedTheta = nnparamsToThetas(newTheta, input_layer_size, hidden_layer_size, hidden_layer_count,
                                                   num_labels)

                    newPrediction = np.mean(predict(shapedTheta, XVal) == yVal) * 100

                    difference = newPrediction - oldIterPred
                    if 0.7 > difference > -0.7 or newPrediction < (2 / num_labels) * 100:
                        break
                    else:
                        oldTheta = newTheta
                        oldIterPred = newPrediction

                print("latest prediction: ", newPrediction)
                if newPrediction > bestPrediction:
                    bestPrediction = newPrediction
                    bestTheta = newTheta
                    bestSize = hidden_layer_size
                    bestCount = hidden_layer_count
                print(f"Best Prediction: {bestPrediction}")
                if bestPrediction >= 90:
                    print(f"it took {time.time() - startTime} seconds to find optimal setting for neural network")
                    return bestTheta, bestSize, bestCount

    print(f"it took {time.time() - startTime} seconds to find optimal setting for neural network")
    return bestTheta, bestSize, bestCount


# works probably
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
        temp[i, int(y[i])] = 1

    a = []
    z = []
    a.append(np.transpose(X))
    for i in range(len(thetas)):
        a[i] = np.concatenate([np.ones((1, a[i].shape[1])), a[i]], axis=0)
        z.append(np.dot(thetas[i], a[i]))
        a.append(sigmoid(z[i]))

    h = np.transpose(a[-1])

    regTerm = 0
    for theta in thetas:
        regTerm += (lambda_ / (2 * m)) * np.sum(theta[:, 1:] ** 2)

    J = (np.sum(-temp * np.log(h + 0.00000001) - ((1 - temp) * np.log(1 - (h - 0.00000001))))) / m + regTerm

    d = []
    thetaGrads = []
    d.append(np.transpose(h - temp))
    for i in range(len(thetas) - 1, 0, -1):
        d.insert(0, np.dot(np.transpose(thetas[i][:, 1:]), d[0]) * sigmoidGradient(z[i - 1]))

    for i in range(len(thetas)):
        thetaGrads.append(np.dot(d[i], np.transpose(a[i])) / m + (lambda_ / m) * np.concatenate(
            [np.zeros((thetas[i][:, 1:].shape[0], 1)), thetas[i][:, 1:]],
            axis=1))

    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([thetaGrad.ravel() for thetaGrad in thetaGrads])
    return J, grad


# I think this works
def optimizeNN(maxIter, input_layer_size, hidden_layer_size,
               hidden_layer_count,
               num_labels, X, y, lambda_, useOtherThetas, initThetas):
    print(f"optimizing network")
    options = {'maxiter': maxIter}

    costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                            hidden_layer_size,
                                            hidden_layer_count,
                                            num_labels, X, y, lambda_)

    start_time = time.time()
    if useOtherThetas:
        initial_nn_params = initThetas
    else:
        initialThetas = randInitAllWeights(input_layer_size, hidden_layer_size, hidden_layer_count, num_labels)
        initial_nn_params = np.concatenate([theta.ravel() for theta in initialThetas], axis=0)

    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)
    print(f"it took {time.time() - start_time} seconds to optimize neural network")
    nn_params = res.x

    return nn_params


# ============basic functions==================
# definitely works(overflows?)
def sigmoid(z):
    gz = 1.0 / (1.0 + np.exp(-z))
    return gz


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

# %%
