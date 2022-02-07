# _*_ coding:utf-8 _*_
# Author: Yuchen Shen
# Name: Manual SVM to achieve image classification
# Date: 2020/7/17
# Time: 22:26

from numpy import *
import numpy as np
from sklearn.decomposition import PCA

class optStruct:
    # Store data with objects
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # Error cache. If 1st is listed as 1, it is valid (error calculated)


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b  # predicted value
    Ek = fXk - float(oS.labelMat[k])  # Error (predicted value minus true value)
    return Ek


def selectJrand(i, m):
    #select one value(i!=j) randomly
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    # Choose j by maximizing the step size (i.e. choose the 2nd alpha)
    maxK = -1
    maxDeltaE = 0  # Used to cache the maximum error, use the smallest possible value as the initial value
    Ej = 0
    oS.eCache[i] = [1, Ei]  # Error buffer. When 1st is listed as 1, it means it is valid (the error is calculated)
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # Returns an array of row indices corresponding to non-zero error buffers
    '''
    m.A Represents the array corresponding to matrix m (matrix to array)
    >>> x = np.array([[1,0,0], [0,2,0], [1,1,0]])
    >>> x
    array([[1, 0, 0],
           [0, 2, 0],
           [1, 1, 0]])
    >>> np.nonzero(x) #Returns a row-indexed array of non-zero elements of the array, and a column-indexed array (a list of components)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
    '''
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # Loop to find the largest delta E
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # validEcacheList is empty，Indicates the first cycle. then randomly select j different from i
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # Update new value to error buffer after any alpha change
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    # alphaj is limited between L and H
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # different from the simplified version
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # update to error cache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print ("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # ai and aj are equal in magnitude
        updateEk(oS, i)  # Update to error buffer, reverse direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    # An implementation of SVM, minimum sequence method (SMO)
    # Full version, but does not use kernel functions, only suitable for dividing basically linearly separable datasets
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter_ = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter_ < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # iterate all values
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                # print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter_ += 1
        else:
            # Returns the row index of the alpha between 0 and C (exclusive)
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]

            #Traverse all non-boundary values ​​(non-0 and non-C)
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter_ += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter_)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):  # Calculate the weight factor
    X = mat(dataArr);
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def img2vector(filename):
    """
   Convert a 32x32 binary image to a 1x1024 vector.
    Parameters:
        filename 
    Returns:
        returnVect - 1x1024 vector of the returned binary image
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def classfy(Xi, w, b):  # make classification predictions
    y = Xi * w + b
    return 1 if y > 0 else -1


def loadImages(dirName, type):
    """
    upload pictures
    Parameters:
        dirName
    Returns:
        trainingMat 
        hwLabels 
    """
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    fileNameStr_list = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == type:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
        fileNameStr_list.append(fileNameStr)
    return trainingMat, hwLabels, fileNameStr_list


def Digits(type):
    dataArr, labelArr, fileNameStr_list = loadImages('trainingDigits', type)
    pca = PCA(n_components=2)
    dataArr = pca.fit_transform(dataArr)
    print(dataArr)
    b, alphas = smoP(dataArr, labelArr, C=0.6, toler=0.001, maxIter=40)
    w = calcWs(alphas, dataArr, labelArr)
    print("Bias-b: \n", b, "\n")
    # print("alphas:\n", alphas,"\n")
    print("eight matrix-w:\n", w, "\n")  # predictive value Y = Xi *w + b
    print("support vector:")
    for i in range(len(alphas)):
        if alphas[i] > 0:  # print support vector
            print(dataArr[i], labelArr[i])
    m, n = mat(dataArr).shape
    Y_predict = zeros(m)
    for i in range(m):
        x = mat(dataArr)[i]
        Y_predict[i] = classfy(x, w, b)
    return Y_predict, fileNameStr_list


# do ont-hot

def one_hot(Y_ALL_TYPE):
    Y_ALL_TYPE = np.stack(Y_ALL_TYPE.tolist(), axis=1)
    Y_ALL_TYPE = Y_ALL_TYPE.tolist()
    Y_ALL_TYPE = [i.tolist() for i in Y_ALL_TYPE]
    results = []
    for type in Y_ALL_TYPE:
        for index, i in enumerate(type):
            if i == 1:
                results.append(index)
                break
    return results


if __name__ == '__main__':
    Y_ALL_TYPE = []
    fileNameStr_list = []
    for i in range(10):
        Y_predict, fileNameStr_list = Digits(type)
        Y_ALL_TYPE.append(Y_predict)
    results = one_hot(Y_ALL_TYPE)

    # write answers in the txt files
    with open("train_result.txt", 'w')  as fp:
        for filename_str, type in zip(fileNameStr_list, results):
            fp.write("{}\t{}\n".format(filename_str, type))
