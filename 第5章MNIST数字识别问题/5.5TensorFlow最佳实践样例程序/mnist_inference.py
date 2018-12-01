# -*- coding: utf-8 -*-

import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularization_rate, regular_flag):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regular_flag == True:
        tf.add_to_collection("losses", tf.nn.l2_loss(weights) * regularization_rate)

    return weights


# 定义神经网络的前向传播过程
def inference(input_tensor, regularization_rate, regular_flag):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularization_rate, regular_flag)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularization_rate, regular_flag)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1, weights) + biases

    #返回最后前向传播的结果
    return layer2





