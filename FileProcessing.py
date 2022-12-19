import os
import time

import numpy as np
from PIL import Image, ImageFile, ImageFilter
from imblearn.over_sampling import RandomOverSampler

import six
import sys
sys.modules['sklearn.externals.six'] = six

ImageFile.LOAD_TRUNCATED_IMAGES = True


def storeFileInFolder(pathToStoreData, dataFileName, data):
    print(f'Storing {dataFileName}')
    if not os.path.isdir(f"{pathToStoreData}"):
        os.mkdir(f"{pathToStoreData}")
    np.savetxt(f'{pathToStoreData}\\{dataFileName}', data, delimiter=',')


def processFile(path_to_file, input_size):
    img = Image.open(path_to_file)
    processedImg = img.resize(input_size).convert('L').filter(ImageFilter.FIND_EDGES)
    processedImg.show()
    file_data = np.asarray(processedImg)
    file_data = file_data.flatten()
    return file_data


def importFiles(pathToDataset, input_size, output_size, training_fraction, validating_fraction, names_list):
    if len(names_list) > 0:
        useList = True
    else:
        useList = False

    start_time = time.time()
    planeDict = {}
    planeCount = {}
    X = []
    y = []
    XVal = []
    yVal = []
    Xtest = []

    for subdir, _, files in os.walk(pathToDataset):
        if subdir == pathToDataset:
            continue
        plane_type = subdir.removeprefix(f'{pathToDataset}\\')
        if useList:
            if plane_type not in names_list:
                continue
        if len(planeDict) == output_size:
            break

        planeDict[plane_type] = len(planeDict)
        planeCount[plane_type] = 0
        for file in files:
            planeCount[plane_type] += 1
            img = Image.open(os.path.join(subdir, file))
            processedImg = img.resize(input_size).convert('L').filter(ImageFilter.FIND_EDGES)
            numpydata = np.asarray(processedImg)

            if (planeCount[plane_type] % 10) < int((training_fraction + validating_fraction) * 10):
                if (planeCount[plane_type] % 10) < int(training_fraction * 10):
                    X.append(numpydata.flatten())
                    y.append(len(planeDict)-1)
                else:
                    XVal.append(numpydata.flatten())
                    yVal.append(len(planeDict)-1)
            else:
                Xtest.append(numpydata.flatten())

    ros = RandomOverSampler(sampling_strategy='not majority', random_state=0)
    reX, reY = ros.fit_resample(X, y)
    reXval, reYval = ros.fit_resample(XVal, yVal)
    X = np.insert(reX, len(reX[0]), reY, axis=1)
    XVal = np.insert(reXval, len(reXval[0]), reYval, axis=1)
    Xtest = np.asarray(Xtest)

    print(planeCount)
    print('It took {:.2f} seconds to import images'.format(time.time() - start_time))
    return X, XVal, Xtest

#%%
