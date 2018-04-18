import numpy as np
import h5py
import tensorflow as tf
from tensorlayer.layers import *
from resnet import *
from init import *
import sys
import plot_trajectory
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()

def variable_summaries(var, name):
    #将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope('summaries'):
        #通过tf.summary.histogram函数记录张量中元素的取值分布。
        tf.summary.histogram(name, var)
        #计算变量的平均值，并定义生成平均信息日志的操作。
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        #计算变量的标准差，并定义生成其日志的操作。
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

def unit_lstm(train, hidden_size): 
    #定义一层LSTM_cell,只需说明hidden_size，它会自动匹配输入的x的维度
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = hidden_size, forget_bias=1.0, state_is_tuple=True)
    #添加dropout layer,一般只设置 output_keep_prob
    if train:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
    else:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    
    return lstm_cell

def resnet_inference(input_tensor, train):
    slim = tf.contrib.slim
    """
    gamma_init = tf.random_normal_initializer(1., 0.02)
    
    relu_1 = tf.nn.relu(slim.conv2d(input_tensor, 32, [5,5]))
    pool_1 = slim.max_pool2d(relu_1, [2,2])
    net_in = InputLayer(pool_1)
    bn_1 = BatchNormLayer(net_in, act=tf.identity, is_train=train, gamma_init=gamma_init)

    net = res_identity(bn_1, 32, (3,3), 'resnet_layer_1', train)
    net = res_identity(net, 32, (3,3), 'resnet_layer_2', train)
    net = res_identity(net, 32, (3,3), 'resnet_layer_3', train)
    net = res_change(net, 64, (3,3), 'resnet_layer_4', train)
    net = res_identity(net, 64, (3,3), 'resnet_layer_5', train)
    net = res_identity(net, 64, (3,3), 'resnet_layer_6', train)
    net = res_identity(net, 64, (3,3), 'resnet_layer_7', train)
    net = res_change(net, 64, (3,3), 'resnet_layer_8', train)
    net = res_identity(net, 64, (3,3), 'resnet_layer_9', train)
    net = res_identity(net, 64, (3,3), 'resnet_layer_10', train)
    net = res_identity(net, 64, (3,3), 'resnet_layer_11', train)
    net.outputs = slim.max_pool2d(net.outputs,[2,2])
    net_out = net.outputs
    """
    net_in = InputLayer(input_tensor)
    net = Conv2d(net_in, 32, (7,7), name='Conv_layer1')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer1')
    net = Conv2d(net, 32, (7,7), name='Conv_layer2')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer2')
    net = MaxPool2d(net, name='pool_layer1')

    net = Conv2d(net, 32, (5,5), name='Conv_layer3')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer3')
    net = Conv2d(net, 32, (5,5), name='Conv_layer4')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer4')
    net = MaxPool2d(net, name='pool_layer2')

    net = Conv2d(net, 64, (3,3), name='Conv_layer5')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer5')
    net = Conv2d(net, 64, (3,3), name='Conv_layer6')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer6')
    net = MaxPool2d(net, name='pool_layer3')
    
    net = Conv2d(net, 64, (3,3), name='Conv_layer7')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer7')
    net = Conv2d(net, 64, (3,3), name='Conv_layer8')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=train, name='bn_layer8')
    net = MaxPool2d(net, name='pool_layer4')

    net_out = net.outputs
    pool_shape = net_out.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

    reshaped_1 = tf.reshape(net_out, [-1, nodes])
    reshaped_2 = tf.reshape(reshaped_1, [-1, 1, nodes])
    print('aaaa',reshaped_1.shape, reshaped_2.shape)
    return reshaped_2

def rnn_inference(input_tensor, batch_size, train, regularizer, dataset):
    x_tensor = input_tensor
    batch_size = batch_size
    input_size = input_tensor.shape[1]    #每个时刻的输入特征
    timestep_size = batch_size            #时序持续长度为batch，即每做一次预测，需要先输入batch_size行
    hidden_size = 640                   #每个隐层的节点数目
    layer_num = 2
    if dataset == 'KITTI':
        class_num = 12                    #输出的维度
    elif dataset == 'TUM':
        class_num = 7
    #调用MultiRNNCell 来实现多层LSTM，下面是2层RNN
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm(train, hidden_size) for i in range(layer_num)],state_is_tuple=True)

    #用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size,dtype = 'float')
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs = x_tensor, initial_state = init_state, time_major=False)
    h_state = outputs[:,-1,:]
    W = weight_variable([hidden_size, class_num])
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(W))
    bias = bias_variable([class_num])
    y_pre = tf.nn.relu(tf.matmul(h_state, W) + bias)
    print('ssss',y_pre.shape)
    """
    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable("weight", [y_pre.shape[1],12], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        fc1_biases = tf.get_variable("bias", [12], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(y_pre, fc1_weights) + fc1_biases)
        if train:
            tf.nn.dropout(fc1, 0.5)
    """
    return y_pre