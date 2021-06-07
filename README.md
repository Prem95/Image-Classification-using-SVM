# ImageClassification

A simple implementation of multi-stage image classification using support vector machine (SVM). Further classification using Feature Vector and Softmax score are done using SVM

(This is not a deployment-ready code as much of the files are redacted)

## Getting Started

Libraries include ````numpy, pandas, tensorflow, matplotlib````

1. alexnetG2.py - Main Alexnet implementation using the pretrained weights from the model zoo. The pretrained weights are related to ImageNet
2. datagenerator.py - On-the-fly data generator for training
3. caffe_classes.py - Contains the classes script for training
4. finetune.py - Main implementation of the whole training pipeline using softmax
5. testing.py - Sample script to perform the model testing using TF1

## Helper Code

````SVM_FeatureVector.ipynb````

- Features from the penultimate layer are extracted and a linear SVM is training for 2 class classification

````SVM_SoftmaxScore.ipynb````

- Final layer scores are extracted to be using in SVM as input. The classes are separated using a hyperplane which is trained according to the distances from the score of the classes.

## Author

Name: Prem Kumar
Date: 7th June 2021

## Help

Please do reach out if more information or any help is required in running these files. Keep in mind that these files are reserved as a guide only.