# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 23:43:20 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import tensorflow as tf
import numpy as np

def cycle_fn(iteration, base_lr, max_lr, stepsize):
    cycle = np.floor(1+iteration/(2*stepsize))
    x = np.abs(iteration/stepsize - 2*cycle +1)
    lr = base_lr + (max_lr - base_lr)*np.maximum(0, (1-x))
    return np.float32(lr)


def cycle_lr(base_lr, max_lr, iter_in_batch, epoch_for_cycle, ratio, total_epochs):
    iteration = 0;
    Lr = [];
    stepsize = (iter_in_batch*epoch_for_cycle)/2.
    for i in range(total_epochs):
        for j in range(iter_in_batch):
            Lr.append(cycle_fn(iteration, base_lr = base_lr, 
                            max_lr = max_lr, stepsize = stepsize))
            iteration+=1
    final_iter = np.int((total_epochs/epoch_for_cycle)*stepsize*2*ratio)
    Lr = np.array(Lr)
    Lr[final_iter:] = base_lr
    return Lr


def Batch_norm(x, training):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        decay=0.997, epsilon=1e-5,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=training, fused=True)
   
def Transformation(x):        
    x = tf.image.resize_image_with_crop_or_pad(
            x, 40, 40)
    x = tf.random_crop(x, [tf.shape(x)[0], 32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x

def Augmentation(x, training):
    x = tf.cond(training, lambda: Transformation(x), lambda: x)
    mean = tf.constant(np.array([0.4914, 0.4822, 0.4465]), tf.float32)
    std = tf.constant(np.array([0.2470, 0.2435, 0.2616]), tf.float32)       
    x = tf.div(tf.subtract(x,mean),std)
    return x


    
class Building_block(object):
    def __init__(self,n_out_filters, training, projection_shortcut, strides):
        self.n_out_filters = n_out_filters   
        self.training = training   
        self.projection_shortcut = projection_shortcut   
        self.strides = strides   

    def Forward(self, x):
        shortcut = x
        x = Batch_norm(x, self.training)
        x = tf.nn.relu(x)
    
        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(x)
    
        x = tf.layers.conv2d(
            inputs=x, filters=self.n_out_filters, kernel_size=3, strides=self.strides,
            padding = 'SAME')
    
        x = Batch_norm(x, self.training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(
            inputs=x, filters=self.n_out_filters, kernel_size=3, strides=1,
            padding = 'SAME')
    
        return x + shortcut

    

class Bottleneck_block(object):
    def __init__(self,n_out_filters, training, projection_shortcut, strides):
        self.n_out_filters = n_out_filters   
        self.training = training   
        self.projection_shortcut = projection_shortcut   
        self.strides = strides   
        
    def Forward(self, x):
        shortcut = x
        x = Batch_norm(x, self.training)
        x = tf.nn.relu(x)
    
        if self.projection_shortcut is not None:
          shortcut = self.projection_shortcut(x)
    
        x = tf.layers.conv2d(
            inputs=x, filters=self.n_out_filters, kernel_size=1, strides=1,
            padding = 'SAME')
    
        x = Batch_norm(x, self.training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(
            inputs=x, filters=self.n_out_filters, kernel_size=3, strides=self.strides,
            padding = 'SAME')
    
        x = Batch_norm(x, self.training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(
            inputs=x, filters=4 * self.n_out_filters, kernel_size=1, strides=1,
            padding = 'SAME')
    
        return x + shortcut
        

class Block_layer(object):
    def __init__(self, n_out_filters, bottleneck, block_fn, blocks, strides,
                 training):
        self.n_out_filters = n_out_filters   
        self.bottleneck = bottleneck   
        self.block_fn = block_fn   
        self.blocks = blocks   
        self.strides = strides   
        self.training = training   
        
        self.Block_seq()

    def projection_shortcut(self, x):
        n_out_filters_shortcut = self.n_out_filters * 4 if self.bottleneck else self.n_out_filters
        return tf.layers.conv2d(inputs=x, filters=n_out_filters_shortcut, kernel_size = 1, 
                         strides = self.strides, padding = 'SAME')

    def Block_seq(self):
        self.block_seq = []
        for _ in range(1, self.blocks):
            block_fn = self.block_fn(self.n_out_filters, self.training, None, 1)
            self.block_seq.append(block_fn)
            
    def Forward(self, x):
        first_block = self.block_fn(self.n_out_filters, self.training, self.projection_shortcut, self.strides)
        
        x = first_block.Forward(x)
        
        for i in range(1, self.blocks):
            x = self.block_seq[i-1].Forward(x)
            
        return x

