import numpy as np
import tensorflow as tf
from tensorlayer.layers import *

slim = tf.contrib.slim
def res_identity(net0, conv_depth, kernel_shape, layer_name, train):
    """不改变输入张量的维度"""
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(layer_name):
        net = Conv2d(net0, conv_depth, kernel_shape, b_init=None, name=layer_name+'/conv_1')
        bn_1 = BatchNormLayer(net, act=tf.nn.relu, is_train= train, gamma_init=gamma_init, name=layer_name+'/bn_1')

        net = Conv2d(bn_1, conv_depth, kernel_shape, b_init=None, name=layer_name+'/conv_2')
        bn_2 = BatchNormLayer(net, act=tf.identity, is_train= train, gamma_init=gamma_init, name=layer_name+'/bn_2')

        net = ElementwiseLayer(layer=[bn_2,net0], combine_fn=tf.add, name=layer_name+'/add')
        net.outputs = tf.nn.relu(net.outputs)
    return net

def res_change(net0, conv_depth, kernel_shape, layer_name, train):
    """改变输入张量的维度"""
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(layer_name):
        net = Conv2d(net0, conv_depth, kernel_shape, strides=(2,2), b_init=None, name=layer_name+'/conv_1')
        bn_1 = BatchNormLayer(net, act=tf.nn.relu, is_train= train, gamma_init=gamma_init, name=layer_name+'/bn_1')

        net0_reshape = Conv2d(net0, conv_depth, (1,1), strides=(2,2), name=layer_name+'/conv_2')
        bn_2 = BatchNormLayer(net0_reshape, act=tf.identity, is_train= train, gamma_init=gamma_init, name=layer_name+'/bn_2')

        net = Conv2d(bn_1, conv_depth, kernel_shape, b_init=None, name=layer_name+'/conv_3')
        bn_3 = BatchNormLayer(net, act=tf.identity, is_train= train, gamma_init=gamma_init, name=layer_name+'/bn_3')

        net = ElementwiseLayer(layer=[bn_3,bn_2], combine_fn=tf.add, name=layer_name+'/add')
        net.outputs = tf.nn.relu(net.outputs)
    return net