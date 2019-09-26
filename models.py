# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 23:41:44 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

resnet model
"""
import tensorflow as tf
from networks import Batch_norm, Augmentation

class ResNet18(object):
    def __init__(self, num_classes, name, shared):
        self.num_classes = num_classes
        self.name = name
        self.shared = shared
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name = "resnet_input")
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_classes), name = "resnet_one_hot")  # onehot
        self.training = tf.placeholder(tf.bool,shape=(), name = "training")
        self.keep_prob = tf.placeholder(tf.float32, name = "drop_out")

    def shared_layers(self, inputs, reuse):
        self.conv1 = tf.layers.conv2d(
          inputs=inputs, filters=64, kernel_size=3,
          strides=2, padding = 'SAME', name = 'conv1', reuse=reuse)
        self.droput1 = tf.nn.dropout(self.conv1, keep_prob= self.keep_prob)
        
        
        ### block 1-1###
        self.shortcut1_1 = self.droput1
        self.conv1 = Batch_norm(self.droput1, self.training)
        self.conv1 = tf.nn.relu(self.conv1)
        self.conv1_1_1 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv1_1_1', reuse=reuse)

        self.conv1_1_1 = Batch_norm(self.conv1_1_1, self.training)
        self.conv1_1_1 = tf.nn.relu(self.conv1_1_1)
        self.conv1_1_2 = tf.layers.conv2d(
            inputs=self.conv1_1_1, filters=64, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv1_1_2', reuse=reuse)
        
        self.block1_1 = self.conv1_1_2 + self.shortcut1_1
        
        ### block 1-2###
        self.shortcut1_2 = self.block1_1
        self.block1_1 = Batch_norm(self.block1_1, self.training)
        self.block1_1 = tf.nn.relu(self.block1_1)
        self.conv1_2_1 = tf.layers.conv2d(
            inputs=self.block1_1, filters=64, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv1_2_1', reuse=reuse)

        self.conv1_2_1 = Batch_norm(self.conv1_2_1, self.training)
        self.conv1_2_1 = tf.nn.relu(self.conv1_2_1)
        self.conv1_2_2 = tf.layers.conv2d(
            inputs=self.conv1_2_1, filters=64, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv1_2_2', reuse=reuse)
        
        self.block1_2 = self.conv1_2_2 + self.shortcut1_2
        

        ### block 2-1###
        self.shortcut2_1 = tf.layers.conv2d(inputs=self.block1_2, filters=128, 
            kernel_size = 1, strides = 2, padding = 'SAME', name = 'shortcut2_1', reuse=reuse)
        self.block1_2 = Batch_norm(self.block1_2, self.training)
        self.block1_2 = tf.nn.relu(self.block1_2)
        self.conv2_1_1 = tf.layers.conv2d(
            inputs=self.block1_2, filters=128, kernel_size=1, strides=2,
            padding = 'SAME', name = 'conv2_1_1', reuse=reuse)
        
        self.conv2_1_1 = Batch_norm(self.conv2_1_1, self.training)
        self.conv2_1_1 = tf.nn.relu(self.conv2_1_1)
        self.conv2_1_2 = tf.layers.conv2d(
            inputs=self.conv2_1_1, filters=128, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv2_1_2', reuse=reuse)
        
        self.block2_1 = self.conv2_1_2 + self.shortcut2_1

        ### block 2-2###
        self.shortcut2_2 = self.block2_1
        self.block2_1 = Batch_norm(self.block2_1, self.training)
        self.block2_1 = tf.nn.relu(self.block2_1)
        self.conv2_2_1 = tf.layers.conv2d(
            inputs=self.block2_1, filters=128, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv2_2_1', reuse=reuse)

        self.conv2_2_1 = Batch_norm(self.conv2_2_1, self.training)
        self.conv2_2_1 = tf.nn.relu(self.conv2_2_1)
        self.conv2_2_2 = tf.layers.conv2d(
            inputs=self.conv2_2_1, filters=128, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv2_2_2', reuse=reuse)
        
        self.block2_2 = self.conv2_2_2 + self.shortcut2_2

  
        ### block 3-1###
        self.shortcut3_1 = tf.layers.conv2d(inputs=self.block2_2, filters=256, 
            kernel_size = 1, strides = 2, padding = 'SAME', name = 'shortcut3_1', reuse=reuse)
        self.block2_2 = Batch_norm(self.block2_2, self.training)
        self.block2_2 = tf.nn.relu(self.block2_2)
        self.conv3_1_1 = tf.layers.conv2d(
            inputs=self.block2_2, filters=256, kernel_size=3, strides=2,
            padding = 'SAME', name = 'conv3_1_1', reuse=reuse)

        self.conv3_1_1 = Batch_norm(self.conv3_1_1, self.training)
        self.conv3_1_1 = tf.nn.relu(self.conv3_1_1)
        self.conv3_1_2 = tf.layers.conv2d(
            inputs=self.conv3_1_1, filters=256, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv3_1_2', reuse=reuse)
        
        self.block3_1 = self.conv3_1_2 + self.shortcut3_1
        
        ### block 3-2###
        self.shortcut3_2 = self.block3_1
        self.block3_1 = Batch_norm(self.block3_1, self.training)
        self.block3_1 = tf.nn.relu(self.block3_1)
        self.conv3_2_1 = tf.layers.conv2d(
            inputs=self.block3_1, filters=256, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv3_2_1', reuse=reuse)

        self.conv3_2_1 = Batch_norm(self.conv3_2_1, self.training)
        self.conv3_2_1 = tf.nn.relu(self.conv3_2_1)
        self.conv3_2_2 = tf.layers.conv2d(
            inputs=self.conv3_2_1, filters=256, kernel_size=3, strides=1,
            padding = 'SAME', name = 'conv3_2_2', reuse=reuse)
        
        self.block3_2 = self.conv3_2_2 + self.shortcut3_2
        return self.block3_2


    def Forward(self):
        self.aug = Augmentation(self.x, self.training)

        self.shared_out = self.shared_layers(self.aug, self.shared)
        ### block 4-1###
        self.shortcut4_1 = tf.layers.conv2d(inputs=self.shared_out, filters=512, 
            kernel_size = 1, strides = 2, padding = 'SAME', name = self.name+'_shortcut4_1')
        self.block3_2 = Batch_norm(self.block3_2, self.training)
        self.block3_2 = tf.nn.relu(self.block3_2)
        self.conv4_1_1 = tf.layers.conv2d(
            inputs=self.block3_2, filters=512, kernel_size=3, strides=2,
            padding = 'SAME', name = self.name+'_conv4_1_1')

        self.conv4_1_1 = Batch_norm(self.conv4_1_1, self.training)
        self.conv4_1_1 = tf.nn.relu(self.conv4_1_1)
        self.conv4_1_2 = tf.layers.conv2d(
            inputs=self.conv4_1_1, filters=512, kernel_size=3, strides=1,
            padding = 'SAME', name = self.name+'_conv4_1_2')
        
        self.block4_1 = self.conv4_1_2 + self.shortcut4_1
        
        ### block 4-2###
        self.shortcut4_2 = self.block4_1
        self.block4_1 = Batch_norm(self.block4_1, self.training)
        self.block4_1 = tf.nn.relu(self.block4_1)
        self.conv4_2_1 = tf.layers.conv2d(
            inputs=self.block4_1, filters=512, kernel_size=3, strides=1,
            padding = 'SAME', name = self.name+'_conv4_2_1')

        self.conv4_2_1 = Batch_norm(self.conv4_2_1, self.training)
        self.conv4_2_1 = tf.nn.relu(self.conv4_2_1)
        self.conv4_2_2 = tf.layers.conv2d(
            inputs=self.conv4_2_1, filters=512, kernel_size=3, strides=1,
            padding = 'SAME', name = self.name+'_conv4_2_2')
        
        self.block4_2 = self.conv4_2_2 + self.shortcut4_2


        self.batch_norm1 = Batch_norm(self.block4_2, self.training)
        self.relu1 = tf.nn.relu(self.batch_norm1)
    
        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
#        with tf.name_scope('mean_pool_output'):
        self.pool = tf.reduce_mean(self.relu1, [1, 2], keepdims=True)
        
        self.reshape = tf.reshape(self.pool, [-1, 1*1*512])

        self.pred = tf.layers.dense(inputs=self.reshape, units=self.num_classes,
                                    name = self.name+'_final_output')
   
#        with tf.name_scope('loss'):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred,
                                                                 labels=tf.stop_gradient([self.y])))

        def Summaray():
            tf.summary.scalar(self.name+'loss', self.loss)

        Summaray()

        return self.pred, self.loss
