import os
import numpy as np
import tensorflow as tf
from alexnetType1 import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime

# Path for the files
train_file = '/home/stroke95/Desktop/PS_Classfier/Path/Train_Bank1.txt'
val_file = '/home/stroke95/Desktop/PS_Classfier/Path/Val_Bank1.txt'

# Hyperparams
learning_rate = 0.01  
num_epochs = 100
batch_size = 32  # batch_size * 227 * 227
dropout_rate = 0.5
num_classes = 2  # Final class output for Diff and Undiff
train_layers = ['fc8', 'fc7']  # Trainable layers, f8 output + f7 feature vector concatenated

# How often we want to write the tf.summary data to disk
display_step = 10

# Path for model checkpoint and tensorboard
tensorboard_path = "/home/stroke95/Desktop/PS_Classfier/tensorboard_SVM"

with tf.device('/cpu:0'):

    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)

    # Create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator
    iterator = iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()

#* Initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)

# <- First batch is ready 32 * 277 * 277
x_input_images = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y_input_labels = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Setting up the environment for SVM
def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variables(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./checkpoint_Stage1/model_epoch500.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_Stage1/'))
    saved_dict = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)   
    model = AlexNet(x_input_images, keep_prob, num_classes, [], saved_dict, load_pretrained_weight = True)

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    # Define activation of last layer as score
    score = model.fc8
    prescore = model.fc7

    prescore_weight = weight_variables([4096, 2])
    prescore_bias = bias_variables([2])
    output = tf.matmul(prescore, prescore_weight) + prescore_bias

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss. Compares the model output with score and input_labels
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = score, labels = y_input_labels))                                                                    
# Train operation
with tf.name_scope("train"):

    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply adam to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + 'Gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary for tensorboard
tf.summary.scalar('Cross Entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y_input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('Accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(tensorboard_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Training now...".format(datetime.now()))
    print("{} --logdir {}".format(datetime.now(), tensorboard_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch : {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x_input_images: img_batch, y_input_labels: label_batch, keep_prob: dropout_rate})
            print(accuracy)