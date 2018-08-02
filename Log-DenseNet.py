# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 22:33:12 2018

@author: agassi001
"""

import tensorflow as tf
import math
from tf.contrib.layers import batch_norm, flatten
from tf.contrib.framework import arg_scope


from tflearn.layers.conv import global_avg_pool

def Conv(x, filters, kernel_size, stride = 1, layer_name="conv"):
    with tf.name_scope(layer_name):
        layer = tf.layers.conv2d(inputs = x, use_bias = False, filters = filters, 
                                   kernel_size = kernel_size, strides = stride, padding='SAME')
    return layer

def Global_Average_Pooling(x, stride=1):
    #Input: Tensor [batch, height, width, in_channels]
    #Outpot: 2-D Tensor [batch, pooled dim]
    return global_avg_pool(x, name ='global_avg_pool')

def Average_Pooling(x, pool_size = [2,2], stride = 2, padding = 'VALID'):
    return tf.layers.average_pooling2d(inputs = x, pool_size = pool_size, strides = stride, padding = padding)

def Max_Pooling(x, pool_size = [2,2], stride = 2, padding = 'VALID'):
    return tf.layers.max_pooling2d(inputs = x, pool_size = pool_size, strides = stride, padding = padding)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs = x, rate = rate, training = training)

def Relu(x):
    return tf.nn.relu(x)

def Concat(layers) :
    return tf.concat(layers, axis=3)

def Batch_Normalization(x, is_training, scope):
    with arg_scope([batch_norm],
                   scope = scope,
                   #decay = 0.9,
                   scale = True,
                   zero_debias_moving_mean = True) :
        return tf.cond(is_training,
                       lambda : batch_norm(inputs = x, is_training = is_training, reuse = None),
                       lambda : batch_norm(inputs = x, is_training = is_training, reuse = True))

def Linear(x, class_num) :
    return tf.layers.dense(inputs = x, units = class_num, name = 'linear')

#dropout_rate = 0.2

class DenseNet():
    def __init__(self, x, n_blocks, filters, is_training):
        self.n_blocks = n_blocks
        self.filters = filters
        self.is_training = is_training
        
    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training = self.training, scope = scope + '_batch1')
            x = Relu(x)
            x = Conv(x, filter=4 * self.filters, kernel = [1,1], layer_name = scope + '_conv1')
            #x = Dropout(x, rate=dropout_rate, training=self.training)
            x = Batch_Normalization(x, training = self.training, scope = scope + '_batch2')
            x = Relu(x)
            x = Conv(x, filter = self.filters, kernel = [3,3], layer_name = scope + '_conv2')
            #x = Dropout(x, rate=dropout_rate, training=self.training)
            return x
        
    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training = self.training, scope = scope + '_batch1')
            x = Relu(x)
            x = Conv(x, filter = self.filters, kernel=[1,1], layer_name = scope + '_conv1')
            #x = Dropout(x, rate=dropout_rate, training=self.training)
            x = Average_Pooling(x, pool_size = [2,2], stride = 2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concat(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concat(layers_concat)
            return x
        
    def log_dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            k = math.floor(math.log(nb_layers, 2))
            
            for i in range(k):
                p2k = 2**i
                x = Concat(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(p2k))
                layers_concat.append(x)

            x = Concat(layers_concat)
            return x
   
    def DenseNet(self, input_x):
        x = Conv(input_x, filter = 2 * self.filters, kernel = [7,7], stride = 2, layer_name = 'conv0')
        
        #DenseNet169
        x = self.dense_block(input_x = x, nb_layers = 6, layer_name = 'dense_1')
        x = self.transition_layer(x, scope = 'trans_1')
        x = self.dense_block(input_x = x, nb_layers = 12, layer_name = 'dense_2')
        x = self.transition_layer(x, scope = 'trans_2')
        x = self.dense_block(input_x = x, nb_layers = 32, layer_name = 'dense_3')
        x = self.transition_layer(x, scope='trans_3')
        x = self.dense_block(input_x = x, nb_layers = 32, layer_name = 'dense_4')
        
        x = Batch_Normalization(x, training = self.training, scope = 'linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        return x