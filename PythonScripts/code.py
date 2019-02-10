from model import model
from utility import processData

############## DEFINING PATHS AND HYPERPARAMETERS ############## 

imgDim = (28,28)
dataDir = '../data/'
flatten = True

dimx, dimy = 392, 392
lmda = 0.02
hLoss = 50
hHiddenDims1, hHiddenDims2 = 500, 300
hDims = 50
lossType = 3

nbEpochs = 10
batchSize = 100

############## EXTRACTING AND PROCESSING DATA ##############

ext = processData(dataDir=dataDir, imgDim=imgDim, flatten=flatten)
[trainImgsLeft, trainImgsRight], trainLabels = ext.extractTrain()

imgs = [trainImgsLeft, trainImgsRight]

leftImgs, rightImgs, labels = ext.trainValidationSplit(imgs, trainLabels)

[trainImgsLeft, validImgsLeft] = leftImgs
[trainImgsRight, validImgsRight] = rightImgs
[trainLabels, validLabels] = labels

############## BUILDING AND TRAINING MODEL ##############

corrNet = model(dimx=dimx,dimy=dimy,lmda=lmda,
                hLoss=hLoss,hDims=hDims,hHiddenDims1=hHiddenDims1,
                hHiddenDims2=hHiddenDims2,lossType=lossType)

endModel, branchModel = corrNet.buildModel()
corrNet.trainModel(endModel, trainImgsLeft, trainImgsRight, 
                   nbEpochs=nbEpochs, batchSize=batchSize)