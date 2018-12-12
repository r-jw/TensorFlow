# -*- coding: utf-8 -*-

import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512


def inference(input_tensor, regularization_rate, regular_flag):
    with tf.variable_scope("layer1_conv1"):
        conv1_weights = tf.get_variable("weight", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("layer2_pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3_conv2"):
        conv2_weights = tf.get_variable("weight", shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, shape=[pool_shape[0], nodes])

    with tf.variable_scope("layer5_fc1"):
        fc1_weights = tf.get_variable("weight", shape=[nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regular_flag == True:
            tf.add_to_collection("losses", tf.nn.l2_loss(fc1_weights) * regularization_rate)

        fc1_biases = tf.get_variable("bias", shape=[FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6_fc2"):
        fc2_weights = tf.get_variable("weight", shape=[FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regular_flag == True:
            tf.add_to_collection("losses", tf.nn.l2_loss(fc2_weights) * regularization_rate)

        fc2_biases = tf.get_variable("bias", shape=[NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit





