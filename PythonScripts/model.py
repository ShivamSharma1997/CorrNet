import numpy as np

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
        
    def buildModel(self):
        inpx = Input(shape=(self.dimx, ))
        inpy = Input(shape=(self.dimy, ))
            
        hx = Dense(self.hHiddenDims1, activation='relu')(inpx)
        hx = Dense(self.hHiddenDims2, activation='relu')(hx)
        hx = Dense(self.hDims, activation='sigmoid')(hx)
        
        hy = Dense(self.hHiddenDims1, activation='relu')(inpy)
        hy = Dense(self.hHiddenDims2, activation='relu')(hy)
        hy = Dense(self.hDims, activation='sigmoid')(hy)
        
        h = add([hx, hy])
        
        recx = Dense(self.dimx, activation='sigmoid')(h)
        recy = Dense(self.dimy, activation='sigmoid')(h)
        
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
        
        model.fit(inp, out, epochs=nbEpochs, batch_size=batchSize, verbose=1)