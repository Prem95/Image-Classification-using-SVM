import os
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime

# Path for the files
train_file = '/home/stroke95/Desktop/PS_Classfier/Path/Train_Bank1.txt'
val_file = '/home/stroke95/Desktop/PS_Classfier/Path/Val_Bank1.txt'

# Hyperparams
learning_rate = 0.001
num_epochs = 500
batch_size = 32  # batch_size * 227 * 227
dropout_rate = 0.5
num_classes = 2  # Final class output for Diff and Undiff
train_layers = ['fc8', 'fc7']  # Trainable layers, f8 output + f7 feature vector concatenated

# How often we want to write the tf.summary data to disk
display_step = 10

# Path for model checkpoint and tensorboard
tensorboard_path = "/home/stroke95/Desktop/PS_Classfier/tensorboard"
checkpoint_path = "/home/stroke95/Desktop/PS_Classfier/checkpoint_testing"

with tf.device('/cpu:0'):

    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)

    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # Create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator
    iterator = iterator.from_structure(tr_data.data.output_types,
                                               tr_data.data.output_shapes)
    next_batch = iterator.get_next()

#* Initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# <- First batch is ready 32 * 277 * 277
x_input_images = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y_input_labels = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize CNN model
model = AlexNet(x_input_images, keep_prob, num_classes, train_layers)

# The final layer for classification
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss. Compares the model output with score and input_labels
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = score,
                                                                     labels = y_input_labels))
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
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

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
            sess.run(train_op, feed_dict={x_input_images: img_batch,
                                          y_input_labels: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x_input_images: img_batch,
                                                        y_input_labels: label_batch,
                                                        keep_prob: 1.}) # Keep the prob as 1 to differentiate it from testing so that all neurons are sensed

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)

        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x_input_images: img_batch,
                                                y_input_labels: label_batch,
                                                keep_prob: 1.})
            test_acc = test_acc + acc
            test_count = test_count + 1
        test_acc /= test_count

        print("{} Validation Accuracy = {:.3f}".format(datetime.now(), test_acc))
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)