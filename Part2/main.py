import numpy as np

rawdata = ""
print "Opening TrainingData"

dataPoints = 0
with open("TrainingData.csv") as fi:
    for line in fi:
        # just ignore lines where we're missing some data from one or more sites
        if (line.count(",,") > 0) or line.count(",") < 27:
            continue
        rawdata += line
        dataPoints += 1

with open("temp_preprocessed.csv", 'w') as fi:
    fi.write(rawdata)

print "Done preprocessing. Loaded", dataPoints, "datapoints. Loading via numpy..."

trainingX = np.loadtxt("temp_preprocessed.csv", delimiter=',', skiprows=1, usecols=(1, 3, 9, 11, 17, 19, 25, 27))
trainingY = np.loadtxt("temp_preprocessed.csv", delimiter=",", skiprows=1, usecols=(5, 7, 13, 15, 21, 23))

print "Done loading."
