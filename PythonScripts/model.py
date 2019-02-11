import numpy as np

from math import sqrt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.layers.merge import add
from keras.layers import Input, Dense

from utility import ZeroPadding, calcCorr, corrLoss

class model:
    def __init__(self, **kwargs):
        self.dimx = kwargs['dimx']
        self.dimy = kwargs['dimy']
        self.lmda = kwargs['lmda']
        self.hLoss = kwargs['hLoss']
        self.hDims = kwargs['hDims']
        self.hHiddenDims1 = kwargs['hHiddenDims1']
        self.hHiddenDims2 = kwargs['hHiddenDims2']
        self.lossType = kwargs['lossType']
        
        self.nbEpochs = kwargs['nbEpochs']
        self.batchSize = kwargs['batchSize']
        self.verbose = kwargs['verbose']
        
    def buildModel(self):
        inpx = Input(shape=(self.dimx, ))
        inpy = Input(shape=(self.dimy, ))
            
        hx = Dense(self.hHiddenDims1, activation='sigmoid')(inpx)
        hx = Dense(self.hHiddenDims2, activation='sigmoid')(hx)
        hx = Dense(self.hDims, activation='sigmoid')(hx)
        
        hy = Dense(self.hHiddenDims1, activation='sigmoid')(inpy)
        hy = Dense(self.hHiddenDims2, activation='sigmoid')(hy)
        hy = Dense(self.hDims, activation='sigmoid')(hy)
        
        h = add([hx, hy])
        
        recx = Dense(self.dimx)(h)
        recy = Dense(self.dimy)(h)
        
        branchModel = Model([inpx, inpy],[recx, recy, h])
        
        recxX, recyX, hX = branchModel([inpx, ZeroPadding()(inpy)])
        recxY, recyY, hY = branchModel([ZeroPadding()(inpx), inpy])
        recxXY, recyXY, hXY = branchModel([inpx, inpy])
        
        corr = calcCorr(-self.lmda)([hX, hY])
        
        inp= [inpx,inpy]
        
        if self.lossType == 1:
            out = [recxX, recxY, recxXY, recyX, recyY, recyXY, corr]
            lossList = ['mse','mse','mse','mse','mse','mse',corrLoss]
        
        elif self.lossType == 2:
            out = [recxX, recxY, recyX, recyY, corr]
            lossList = ['mse','mse','mse','mse',corrLoss]
        
        elif self.lossType == 3:
            out = [recxX, recxY, recxXY, recyX, recyY, recyXY]
            lossList = ['mse','mse','mse','mse','mse','mse']
        
        elif self.lossType == 4:
            out = [recxX, recxY, recyX, recyY]
            lossList = ['mse','mse','mse','mse']
        
        model = Model(inp, out)
        model.compile(loss=lossList, optimizer='rmsprop')
        
        return model, branchModel
    
    def trainModel(self, model, leftData, rightData, nbEpochs=10, batchSize=100):
        corrShape = (leftData.shape[0], self.hLoss)
        corrOut = np.zeros(corrShape)
        
        if self.lossType == 1:
            print('Loss Type: L1 + L2 + L3 - L4')
            print('hDims:', self.hDims)
            print('lamda:', self.lmda)
            out = [leftData, leftData, leftData, rightData, rightData, rightData, corrOut]
        
        elif self.lossType == 2:
            print('Loss Type: L2 + L3 - L4')
            print('hDims:', self.hDims)
            print('lamda:', self.lmda)
            out = [leftData, leftData, rightData, rightData, corrOut]
        
        elif self.lossType == 3:
            print('Loss Type: L1 + L2 + L3')
            print('hDims:', self.hDims)
            print('lamda:', self.lmda)
            out = [leftData, leftData, leftData, rightData, rightData, rightData]
        
        elif self.lossType == 4:
            print('Loss Type: L2 + L3')
            print('hDims:', self.hDims)
            print('lamda:', self.lmda)
            out = [leftData, leftData, rightData, rightData]
        
        inp = [leftData, rightData]
        
        model.fit(inp, out, 
                  epochs=self.nbEpochs, batch_size=self.batchSize,
                  verbose=self.verbose)

class testModel:
    def __init__(self, **kwargs):
        self.leftData = kwargs['leftData']
        self.rightData = kwargs['rightData']
        self.labels = kwargs['labels']
        
        self.model = kwargs['model']
        
        self.sumCorr()
        self.transfer()
    
    def predictHValue(self, inp):
        pred = self.model.predict([inp[0], inp[1]])
        return pred[2]
    
    def sumCorr(self):
        view1 = self.leftData
        view2 = self.rightData
        
        x = self.predictHValue([view1, np.zeros_like(view1)])
        y = self.predictHValue([np.zeros_like(view2), view2])
        
        print('Test Correlation:')
        
        corr = 0
        for i in range(0, x.shape[1]):
            x1 = x[:,i] - (np.ones(len(x))*(sum(x[:,i])/len(x)))
            x2 = y[:,i] - (np.ones(len(y))*(sum(y[:,i])/len(y)))
            corr += sum(x1-x2)/(sqrt(sum(x1*x1))*sqrt(sum(x2*x2)))
        
        print(corr)
    
    def svmClassifier(self, trainX, trainY, testX, testY):
        clf = LinearSVC()
        clf.fit(trainX, trainY)
        pred = clf.predict(testX)
        score = accuracy_score(testY, pred)
        
        return score
    
    def transfer(self):
        view1 = self.leftData
        view2 = self.rightData
        labels = self.labels
        
        view1 = self.predictHValue([view1, np.zeros_like(view1)])
        view2 = self.predictHValue([np.zeros_like(view2), view2])
        
        val = int(len(view1)/5)
        
        print('Transfer Accuracy view1 to view2:')
        
        score = 0
        for i in range(0, 5):
            testX = view2[i*val:(i+1)*val]
            testY = labels[i*val:(i+1)*val]
            
            if i == 0:
                trainX = view1[val:]
                trainY = labels[val:]
            elif i == 4:
                trainX = view1[:4*val]
                trainY = labels[:4*val]    
            else:
                trainX1 = view1[:i*val]
                trainY1 = labels[:i*val]
                
                trainX2 = view1[(i+1)*val:]
                trainY2 = labels[(i+1)*val:]
                
                trainX = np.concatenate((trainX1, trainX2))
                trainY = np.concatenate((trainY1, trainY2))
            
            score += self.svmClassifier(trainX, trainY, testX, testY)
        
        print(score)
        
        print('Transfer Accuracy view2 to view1:')
        
        score = 0
        for i in range(0, 5):
            testX = view1[i*val:(i+1)*val]
            testY = labels[i*val:(i+1)*val]
            
            if i == 0:
                trainX = view2[val:]
                trainY = labels[val:]
            elif i == 4:
                trainX = view2[:4*val]
                trainY = labels[:4*val]    
            else:
                trainX1 = view2[:i*val]
                trainY1 = labels[:i*val]
                
                trainX2 = view2[(i+1)*val:]
                trainY2 = labels[(i+1)*val:]
                
                trainX = np.concatenate((trainX1, trainX2))
                trainY = np.concatenate((trainY1, trainY2))
            
            score += self.svmClassifier(trainX, trainY, testX, testY)
        
        print(score)