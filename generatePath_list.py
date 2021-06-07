import os
import numpy as np
import matplotlib.image as mpimg
import glob
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the classes. See phone image for the other classes
Class_List = [0, 3]
Train_List = [40, 164]
Val_List = [16, 16]
Test_List = [16, 16]
Train_No = 0
Valid_No = 0
Test_No = 0

# Check for files, else create new
os.remove('PKTrain1.txt') if os.path.exists('PKTrain1.txt') else None
os.remove('PKVal1.txt') if os.path.exists('PKVal1.txt') else None
os.remove('PKTest1.txt') if os.path.exists('PKTest1.txt') else None

# Enumerate uses counter so easier
for index, classindex in enumerate(Class_List):

    # Checking the respective class names with enumerate
    className = 'Class' + str(classindex)
    Col1 = []  # Class
    Col2 = []  # Path

    rootClass = '/home/stroke95/Desktop/PS_img_classifier_1'
    for root, dirs, files in os.walk(rootClass, topdown=False):
        if className in root:  
            for name in files:
                path = os.path.join(root, name)
                path = path[1:]
                Col1.append(int(name[0]))
                Col2.append("/" + path)

    # Total number of item in array
    Valid_st = Val_List[index]
    Train_st = Valid_st + Test_List[index]
    Train_stp = Train_st + Train_List[index]

    temp_d = {'Class': Col1, 'Path': Col2}
    Dataframe = pd.DataFrame(data = temp_d)

    # Open, write and auto close
    with open('PKTrain1.txt', 'a') as train, open('PKTest1.txt', 'a') as test, open('PKVal1.txt', 'a') as valid:

        for i in [classindex]:
            classNumber = Dataframe
            testData = classNumber.iloc[0:Valid_st]
            validData = classNumber.iloc[Valid_st:Train_st]
            trainData = classNumber.iloc[Train_st:Train_stp]

            for counter in range(0, len(testData)):
                test.write("{} 0\n".format(
                    testData.Path.values[counter], testData.Class.values[counter]))

            for counter in range(0, len(validData)):
                valid.write("{} 0\n".format(
                    validData.Path.values[counter], validData.Class.values[counter]))

            for counter in range(0, len(trainData)):
                train.write("{} 0\n".format(
                     trainData.Path.values[counter], trainData.Class.values[counter]))                   