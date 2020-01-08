from idxReader import readIdxFile

trainX = readIdxFile('Data/train-images-idx3-ubyte')
trainY = readIdxFile('Data/train-images-idx3-ubyte')

testX = readIdxFile('Data/t10k-images-idx3-ubyte')
testX = readIdxFile('Data/t10k-labels-idx1-ubyte')


