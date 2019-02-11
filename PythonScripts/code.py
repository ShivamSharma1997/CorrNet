from utility import processData
from model import model, testModel

############## DEFINING PATHS AND HYPERPARAMETERS ############## 

imgDim = (28,28)
dataDir = '../data/'
flatten = True

dimx, dimy = 392, 392
lmda = 0.02
hLoss = 50
hHiddenDims1, hHiddenDims2 = 500, 300
hDims = 50
lossType = 2

nbEpochs = 40
batchSize = 100
verbose = 2

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
                hHiddenDims2=hHiddenDims2,lossType=lossType,
                nbEpochs=nbEpochs, batchSize=batchSize,verbose=verbose)

endModel, branchModel = corrNet.buildModel()
corrNet.trainModel(endModel, trainImgsLeft, trainImgsRight)

############## TESTING MODEL ##############

test = testModel(leftData=validImgsLeft, rightData=validImgsRight,
                 labels=validLabels, model=branchModel)