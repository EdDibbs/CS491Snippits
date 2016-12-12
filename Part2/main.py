import numpy as np
import matplotlib.dates as mdates
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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
    Y = []
    
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
    
    if (isTrainData):
        x = np.loadtxt(outputFile, delimiter=',', usecols=colsX, converters={0:date_converter})
        Y = np.loadtxt(outputFile, delimiter=",", usecols=(colsY[1],))
    else:
        x = np.loadtxt(outputFile, delimiter=",", usecols=(1,))
        
    print ("Done preprocessing ", inputFile)
      
    return x, Y

print ("Opening TrainingData")
trainingX, trainingY = preprocess("TrainingData.csv", True)
X_train, X_test, y_train, y_test = train_test_split(trainingX, trainingY, test_size=0.4, random_state=0)
print ("Done loading.")

print ("Opening TestData")
#testX = preprocess("TestData.csv", False) #doesn't work currently
print ("Done loading.")

classifiers = {}
#classifiers["SVC"] = svm.SVC()
classifiers["SVR"] = svm.SVR()
#classifiers["Linear SVC"] = svm.LinearSVC()
#classifiers["Linear SVR"] = svm.LinearSVR()

for label, classifier in classifiers.items():
    
    print(label, "Fitting...");
    clf = classifier
    clf.fit(X_train, y_train)
    print("Validating...")
    predict = clf.predict(X_test)
    mse = mean_squared_error(y_test, predict)
    print ("MSE: ", mse)
    scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    print ("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

