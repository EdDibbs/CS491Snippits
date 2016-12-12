import numpy as np
import matplotlib.dates as mdates
from sklearn import svm

# siteNameDictionary maps col# to Dictionary Name
siteNameDictionary = {}

def strpdate2num(fmt):
    def converter(b):
        return mdates.strpdate2num(fmt)(b.decode('ascii'))
    return converter

def preprocess(inputFile, isTrainData):
    rawdata = ""
    outputFile = "temp_" + inputFile
    x = []
    y1 = []
    y2 = []
    y3 = []
    
    with open(inputFile) as fi:
        # outputDictionary maps '(dateTimeStamp)' to measures of all sites wind velocity
        for lineIndex, line in enumerate(fi):
            if lineIndex == 0 and isTrainData:
                # we're on the first line, let's parse out site names
                for index, col in enumerate(line.split(',')):
                    if col == '':
                        continue
                    if col == "Site Name:":
                        continue
                    
                    siteNameDictionary[index] = col
                
                continue # done parsing site names, continue to the next line
                
            if lineIndex < 9:
                continue
            
            # just ignore lines where we're missing some data from one or more sites
            if (line.count(",,") > 0) or line.count(",") < 14:
                continue
            
            rawdata += line                
                
        
    with open(outputFile, 'w') as fi:
        fi.write(rawdata)

    date_converter = strpdate2num('%m/%d/%Y %H:%M:%S %p')
    colsX = []
    colsY = []
      
    
    if (isTrainData):
        colsX.append(0) #datetime
        for col, site in siteNameDictionary.items():
            if site == "Snake Range West Pinyon-Juniper" or site == "Snake Range West Subalpine" or site == "Snake Range West Montane":
                colsY.append(col)
            else:
                colsX.append(col)
        print ("Using the following cols for X: ", colsX)
        print ("Using the following cols for Y: ", colsY)
        colsX = tuple(colsX)
        colsY = tuple(colsY)
        
    else: # test data
        colsX = tuple(0,)
        
    x = np.loadtxt(outputFile, delimiter=',', usecols=colsX, converters={0:date_converter})
    
    if (isTrainData):
        Y = np.loadtxt(outputFile, delimiter=",", usecols=colsY)
        
    print ("Done preprocessing ", inputFile)
      
    return x, Y

print ("Opening TrainingData")
trainingX, trainingY = preprocess("TrainingData.csv", True)
print ("Done loading.")

print ("Opening TestData")
#testX = preprocess("TestData.csv", False) #doesn't work currently
print ("Done loading.")

print("SVM Fitting...");
clf = svm.SVR()
clf.fit(trainingX, trainingY)  
print("Done.");

