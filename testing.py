import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from alexnetType1 import AlexNet
from caffe_classes import class_names
import shutil

num_classes = 2
original_class_label = ''
miss = 0
good = 0

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = './Path/Test_Bank1.txt'
image_dir = os.path.join(current_dir, 'images')
img_path_list = []

with open(current_dir, 'r') as img_files:

    imgs = []
    test_labels = []
    for i, f in enumerate (img_files):
        img_path, lable = f.split(' ')
        img_path_list.append(img_path)
        imgs.append(plt.imread(img_path))
        test_labels.append(lable)

resultsFilename = './Result_Classifier_Bank1.txt'

x = tf.placeholder(tf.float32, [1, 227, 227, 3]) # Placeholders to store the values for the variables
keep_prob = tf.placeholder(tf.float32)

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./checkpoint_Stage1/model_epoch500.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_Stage1/'))
    saved_dict = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)   
    model = AlexNet(x, keep_prob, num_classes, [], saved_dict, load_pretrained_weight = True)

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    # Define activation of last layer as score
    score = model.fc8
    
    # Create op to calculate softmax 
    softmax = tf.nn.softmax(score)
    
    # Loop over all images
    for i, image in enumerate(imgs):
            
            img = cv2.resize(image.astype(np.float32), (227, 227))
            img = img - imagenet_mean
            img = img.reshape((1, 227, 227, 3))
            probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
            maxprob = np.argmax(probs)
            class_name = class_names[maxprob]
            original_class_label = class_name[6]
            
            # Counter for prediction
            if ((test_labels[i][0] ) != original_class_label): 
                miss = miss + 1

            else:
                good = good + 1
            
            # Filename to write
            with open(resultsFilename, 'a') as myfile:
                myfile.write(img_path_list[i] + 'Predicted: ' + class_name[6] + " %.3f " %probs[0, np.argmax(probs)] + 'Actual: ' + test_labels[i])
    print('Classifier 1: Wrong prediction: ' + str(miss), 'Correct prediction: ' + str(good))
    