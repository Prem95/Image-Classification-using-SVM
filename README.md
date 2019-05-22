# ImageClassification

A simple implementation of multi-stage image classification using support vector machine (SVM). Further classification using Feature Vector and Softmax score are done using SVM

(This is not a deployment-ready code as much of the files are redacted)

## Getting Started

Libraries include ````numpy, pandas, tensorflow, matplotlib````

1. This repository only contains the final stage of a multi-stage classification, which is classification using SVM by Softmax and Feature Vector

2. Initial training checkpoint model and image files are not included

3. Tested on Tensorflow 1.13.

## Main Code

````SVM_FeatureVector.ipynb````

- Features from the penultimate layer are extracted and a linear SVM is training for 2 class classification

````SVM_SoftmaxScore.ipynb````

- Final layer scores are extracted to be using in SVM as input. The classes are separated using a hyperplane which is trained according to the distances from the score of the classes.

# Result 

The confusion matrix displayed below with accuracy of 93% using SVM

![Confusion Matrix](https://raw.githubusercontent.com/Prem95/ImageClassification/master/Screenshot%202019-05-22%20at%208.18.01%20PM.png)
