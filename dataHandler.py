
from pickle import load, dump

def saveDataToFile(data, filePath):
    f = open(filePath, 'wb')
    dump(data, f)
    f.close()

def loadDataFromFile(filePath):
    f = open(filePath, 'rb')
    data = load(f)
    f.close()
    return data

