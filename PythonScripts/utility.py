import numpy as np
import keras.backend as K

from mnist import MNIST
from keras.layers import Layer
from sklearn.model_selection import train_test_split

class processData:
    
    def __init__(self, dataDir='../data/', imgDim=(28,28), flatten=True):

        self.dataDir = dataDir
        self.imgDim = imgDim
        self.flatten = flatten

#Function To Extract Training Data
    def extractTrain(self):
        
        mndata = MNIST(self.dataDir)
        trainImgs, trainLabels = mndata.load_training()

        trainImgs = [np.array(img).reshape(self.imgDim) for img in trainImgs]

        trainImgsLeft, trainImgsRight = [], []
        midVal = int(self.imgDim[0]/2)

        for img in trainImgs:
            if self.flatten:
                trainImgsLeft.append(img[:, :midVal].flatten())
                trainImgsRight.append(img[:, midVal:].flatten())
            else:
                trainImgsLeft.append(img[:, :midVal])
                trainImgsRight.append(img[:, midVal:])

        return [trainImgsLeft, trainImgsRight], trainLabels

#Function To Extract Testing Data        
    def extractTest(self):

        mndata = MNIST(self.dataDir)
        testImgs, testLabels= mndata.load_testing()

        testImgs = [np.array(img).reshape(self.imgDim) for img in testImgs]

        testImgsLeft, testImgsRight = [], []
        midVal = int(self.imgDim[0]/2)

        for img in testImgs:
            testImgsLeft.append(img[:, :midVal])
            testImgsRight.append(img[:, midVal:])

        return [testImgsLeft, testImgsRight], testLabels

#Function To Split Training Data Into Training And Validation Data    
    def trainValidationSplit(self, imgs, labels, testSize = 0.20):
        

        [leftImgs, rightImgs] = imgs
        
        trainImgsLeft, validImgsLeft = train_test_split(leftImgs, 
                                                        test_size=testSize, 
                                                        random_state=11)
        
        trainImgsRight, validImgsRight = train_test_split(rightImgs, 
                                                          test_size=testSize, 
                                                          random_state=11)

        trainLabels, validLabels = train_test_split(labels, 
                                                    test_size=testSize, 
                                                    random_state=11)

        leftImgs = [np.array(trainImgsLeft), np.array(validImgsLeft)]
        rightImgs = [np.array(trainImgsRight), np.array(validImgsRight)]
        labels = [np.array(trainLabels), np.array(validLabels)]
        
        return leftImgs, rightImgs, labels

#Layer To Construct 0-Vector
class ZeroPadding(Layer):
    def __init__(self, **kwargs):
        super(ZeroPadding, self).__init__(**kwargs)

    def call(self, x):
        return K.zeros_like(x)

#Layer To Calculate The Correlation
class calcCorr(Layer):
    def __init__(self, lmda, **kwargs):
        super(calcCorr, self).__init__(**kwargs)
        self.lmda = lmda
    
    def corr(self, x, y):
        xMean = K.mean(x, axis=0)
        yMean = K.mean(y, axis=0)
        
        xCentered = x - xMean
        yCentered = y - yMean
        
        corrN = K.sum(xCentered * yCentered, axis=0)
        
        corrD1 = K.sqrt(K.sum(xCentered * xCentered, axis=0))
        corrD2 = K.sqrt(K.sum(yCentered * yCentered, axis=0))
        
        corrD = corrD1 * corrD2
        
        corr = corrN/corrD
        
        return K.sum(corr) * self.lmda
    
    def call(self, x):
        hX, hY = x[0], x[1]
        corr = self.corr(hX, hY)
        return corr

#Layer To Calculate Correlation Loss
def corrLoss(yTrue, yPred):
    return yPred