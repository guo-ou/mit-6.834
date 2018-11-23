# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:27:09 2018

@author: GuoOu
"""

# Import MINST data
import matplotlib.pyplot as plt
import input_data
mnist = input_data.read_data_sets("./", one_hot=True)
import tensorflow as tf
import numpy as np

def display_minst_image(im):
    im_reshaped = im.reshape((28, 28))
    plt.imshow(im_reshaped, cmap='Greys', interpolation="nearest")
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    
def get_label_from_one_hot(onehot):
    return np.argmax(onehot)

# Show the 8th image (index 7) from the 
# training data set. There's also a validation training set
# to check performance after training.
#display_minst_image(mnist.train.images[7])
#print ("Actual value: {}".format(get_label_from_one_hot(mnist.train.labels[7])))
# Parameters
learning_rate = 0.001
training_iters = 100000
# decsion the Batch size for the SGD algorithm
batch_size = 128
#how often do we show the output
display_step = 20
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input. These "placeholders" will be filled later with the MNIST data set images and ground truth labels
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

##define covolution, ReLU layer and max pooling layer
#def conv2d(img, w, b):
#    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], 
#                                                  padding='VALID'),b))
#
#def max_pool(img, k):
#    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
#
#
#def conv_net(_X, _weights, _biases, _dropout):
#    # The input layer is a 28x28x1 volume (only one color channel - grayscale)
#    # Reshape input picture
#    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
#    # Convolution Layer
#    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
#    # Max Pooling (down-sampling) with a stride of k=3
#    conv1 = max_pool(conv1, k=3)
#    # Apply Dropout (doesn't change number of neurons)
#    conv1 = tf.nn.dropout(conv1, _dropout)
#    
#    # Output volume of this first convolutional layer: 
#    #    Input image 28x28, convolution size of 5x5 yields a 24x24 result (28-5+1)
#    #    we are using 32 activation layers, yielding a resulting output volume after
#    #    the conv2d of 24x24x32.
#    #
#    #    The max pooling layer with a 3x3 stride takes each 3x3 tile of the image
#    #    and yields just a single pixel. So after applying this to each activation layer,
#    #    we result in an 8x8x32 output volume.
#    #
#    #    The dropout above doesn't change the size of the volume, so we finally end up with
#    #    a 8x8x32 output volume after the above convolution, maxpool, and dropout layer.
#
#    # Later on, you'll add another convolutional layer here...
#    
#    
#    
#    # Next, we'll add a fully connected layer from each of these 8x8x32 neurons
#    # to 1024 neurons.
#    # Reshape conv2 output to fit dense layer input
#    dense1 = tf.reshape(conv1, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
#    # Relu activation
#    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
#    # Apply Dropout
#    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout
#
#    # Finally, 10 more neurons fully connected to each of these 1024 above 
#    # for the final class prediction.
#    # Output, class prediction
#    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
#    return out
#
#
## Store layers weight & bias
#weights = {
#    # 5x5 conv, 1 input, 32 outputs
#    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
#    # fully connected, 8*8*32 inputs, 1024 outputs
#    'wd1': tf.Variable(tf.random_normal([8*8*32, 1024])), 
#    # 1024 inputs, 10 outputs (class prediction)
#    'out': tf.Variable(tf.random_normal([1024, n_classes])) 
#}
#
#biases = {
#    'bc1': tf.Variable(tf.random_normal([32])),
#    'bd1': tf.Variable(tf.random_normal([1024])),
#    'out': tf.Variable(tf.random_normal([n_classes]))
#}
#
## Construct model
#pred = conv_net(x, weights, biases, keep_prob)
## Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
## Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
## Initializing the variables
#init = tf.initialize_all_variables()
#saver = tf.train.Saver()
#
## Train the model and get the accuracy score
#with tf.Session() as sess:
#    sess.run(init)
#    step = 1
#    # Keep training until reach max iterations
#    while step * batch_size < training_iters:
#        if step == 1:
#            print ("Training started:")
#        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#        # Fit training using batch data
#        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
#        
#        if step % display_step == 0:
#            # Calculate batch accuracy
#            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
#            # Calculate batch loss
#            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
#            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
#        step += 1
#    print ("Optimization Finished!")
#    # Calculate accuracy for 256 mnist test images
#    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.validation.images[:256], 
#                                                             y: mnist.validation.labels[:256], 
#                                                             keep_prob: 1.}))
#    # Save the model as a file so we can restore it later.
#    saver.save(sess, "/tmp/model_1.ckpt")
#    sess.close()
#
### Steve added this
#pred_out = tf.argmax(pred, 1)
#
#def show_validation_result(sess, i):
#    # Get the prediction and actual label
#    cnn_predicted_digit = sess.run(pred_out, feed_dict={x: np.array([mnist.validation.images[i]]), 
#                                                        keep_prob: 1.})
#    actual_digit = get_label_from_one_hot(mnist.validation.labels[i])
#    display_minst_image(mnist.validation.images[i])
#    plt.title('Predicted: {}, Actual: {}'.format(cnn_predicted_digit, actual_digit), 
#              color=('blue' if actual_digit == cnn_predicted_digit else 'red'))
#    
#def show_many_validation_results(model_filename):
#    with tf.Session() as sess:
#        # Load the stored model
#        saver.restore(sess, model_filename)
#        N = 6
#        M = 12
#        index = 1
#        # Make lots of subplots
#        plt.figure(figsize=(12, 24))
#        for i in range(N):
#            for j in range(M):
#                plt.subplot(M, N, index)
#                show_validation_result(sess, index)
#                index += 1
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#        sess.close()
#
## Might take a few seconds to show up
#show_many_validation_results("/tmp/model_1.ckpt")
#

##define covolution, ReLU layer and max pooling layer
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], 
                                                  padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    # Relu activation
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    # Apply Dropout
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), 
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), 
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes])) 
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Train the model and get the accuracy score
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.validation.images[:256], 
                                                             y: mnist.validation.labels[:256], 
                                                             keep_prob: 1.}))
    # Save the model as a file so we can restore it later.
    saver.save(sess, "/tmp/model_2.ckpt")
    sess.close()
    
## Steve added this
pred_out = tf.argmax(pred, 1)

def show_validation_result(sess, i):
    # Get the prediction and actual label
    cnn_predicted_digit = sess.run(pred_out, feed_dict={x: np.array([mnist.validation.images[i]]), 
                                                        keep_prob: 1.})
    actual_digit = get_label_from_one_hot(mnist.validation.labels[i])
    display_minst_image(mnist.validation.images[i])
    plt.title('Predicted: {}, Actual: {}'.format(cnn_predicted_digit, actual_digit), 
              color=('blue' if actual_digit == cnn_predicted_digit else 'red'))
    
def show_many_validation_results(model_filename):
    with tf.Session() as sess:
        # Load the stored model
        saver.restore(sess, model_filename)
        N = 6
        M = 12
        index = 1
        # Make lots of subplots
        plt.figure(figsize=(12, 24))
        for i in range(N):
            for j in range(M):
                plt.subplot(M, N, index)
                show_validation_result(sess, index)
                index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        sess.close()
        
# Might take a few seconds to show up
show_many_validation_results("/tmp/model_2.ckpt")