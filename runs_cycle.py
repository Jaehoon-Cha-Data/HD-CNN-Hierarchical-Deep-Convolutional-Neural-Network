# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:32:07 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

resnet runs
"""
import tensorflow as tf
from cifar100 import Cifar100
from collections import OrderedDict
from networks import cycle_lr
import argparse
import os
import numpy as np
from models import ResNet18
np.random.seed(0)
tf.set_random_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    # optim config
    parser.add_argument('--model_name', type=str, default = 'ResNet18_200_200_200')
    parser.add_argument('--datasets', type = str, default = 'CIFAR100')
    parser.add_argument('--epochs_set', type=list, default=[200,200,200])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type = list, default = [0.08, 0.008])
    parser.add_argument('--max_lr', type = list, default = [0.5, 0.05])
    parser.add_argument('--cycle_epoch', type = int, default = 20)
    parser.add_argument('--cycle_ratio', type = float, default = 0.7)
    parser.add_argument('--num_fines', type = int, default = 100)
    parser.add_argument('--num_coarse', type = int, default = 20)

    args = parser.parse_args()
    
    config = OrderedDict([
        ('model_name', args.model_name),
        ('datasets', args.datasets),
        ('epochs_set', args.epochs_set),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('max_lr', args.max_lr),
        ('cycle_epoch', args.cycle_epoch),
        ('cycle_ratio', args.cycle_ratio),
        ('num_fines', args.num_fines),
        ('num_coarse', args.num_coarse)])

    return config


config = parse_args()


### call data ###
cifar100 = Cifar100()
n_samples = cifar100.num_examples 
n_test_samples = cifar100.num_test_examples

### make models ###
### shared models ###
model = ResNet18(config['num_fines'],'shared', False)
shared_pred, shared_loss = model.Forward()
### coarse models ###
model_coarse = ResNet18(config['num_coarse'], 'coarse', True)
coarse_pred, coarse_loss = model_coarse.Forward()
### fine models ###
model_fines = {}
fines_pred = {}
fines_loss = {}
for i in range(config['num_coarse']):
    model_fines[i] = ResNet18(config['num_fines'], 'fine{:02}'.format(i), True)
    fines_pred[i], fines_loss[i] = model_fines[i].Forward()
    

### make folder ###
mother_folder = config['model_name']
try:
    os.mkdir(mother_folder)
except OSError:
    pass    


def run(mod, pred, loss, epochs, base_lre, max_lr, name):
    train_loss_set = []
    train_acc_set = []
    test_loss_set = []
    test_acc_set = []
    iter_per_epoch = int(n_samples/config['batch_size'])
    Lr = cycle_lr(base_lre, max_lr, iter_per_epoch, 
              config['cycle_epoch'], config['cycle_ratio'], epochs)

    cy_lr = tf.placeholder(tf.float32, shape=(),  name = "cy_lr_"+name)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cy_lr).minimize(loss)

    prediction = tf.argmax(pred, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(mod.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy_"+name)

    iteration = 0
    iter_per_test_epoch = n_test_samples/config['batch_size'] 
    for epoch in range(epochs):
        epoch_loss = 0.
        epoch_acc = 0.
        for iter_in_epoch in range(iter_per_epoch):
            epoch_x, epoch_fine, epoch_coarse = cifar100.next_train_batch(config['batch_size'])
            _, c, acc = sess.run([optimizer, loss, accuracy],#, summ], 
                            feed_dict = {mod.x: epoch_x, mod.y: epoch_fine, 
                                         mod.training:True, cy_lr: Lr[iteration],
                                         mod.keep_prob: 0.7})
            epoch_loss += c
            epoch_acc += acc
            iteration+=1
            if iter_in_epoch%100 == 0:
                print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
                      'completed out of ', epochs, 'loss: ', epoch_loss/(iter_in_epoch+1),
                      'acc: ', '{:.2f}%'.format(epoch_acc*100/(iter_in_epoch+1)))
        print('######################')        
        print('TRAIN')        
        print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
              'completed out of ', epochs, 'loss: ', epoch_loss/int(iter_per_epoch),
              'acc: ', '{:.2f}%'.format(epoch_acc*100/int(iter_per_epoch)))
        train_loss_set.append(epoch_loss/int(iter_per_epoch))
        train_acc_set.append(epoch_acc*100/int(iter_per_epoch))
        test_loss = 0.
        test_acc = 0.
        for iter_in_epoch in range(int(iter_per_test_epoch)):            
            epoch_x, epoch_fine, epoch_coarse = cifar100.next_test_batch(config['batch_size'])
            c, acc = sess.run([loss, accuracy], 
                              feed_dict = {mod.x: epoch_x, mod.y: epoch_fine, 
                                           mod.training:False, mod.keep_prob:1.})
            test_loss += c
            test_acc += acc
        print('TEST')        
        print('Epoch ', epoch,  'loss: ', test_loss/int(iter_per_test_epoch), 
              'acc: ', '{:.2f}%'.format(test_acc*100/int(iter_per_test_epoch)))
        print('###################### \n')     
        test_loss_set.append(test_loss/int(iter_per_test_epoch))
        test_acc_set.append(test_acc*100/int(iter_per_test_epoch))
        
    return train_loss_set, train_acc_set, test_loss_set, test_acc_set


def run_coarse(mod, pred, loss, epochs, base_lre, max_lr, name, var):
    train_loss_set = []
    train_acc_set = []
    test_loss_set = []
    test_acc_set = []
    iter_per_epoch = int(n_samples/config['batch_size'])
    Lr = cycle_lr(base_lre, max_lr, iter_per_epoch, 
              config['cycle_epoch'], config['cycle_ratio'], epochs)

    cy_lr = tf.placeholder(tf.float32, shape=(),  name = "cy_lr_"+name)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cy_lr).minimize(loss, var_list = var)

    prediction = tf.argmax(pred, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(mod.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy_"+name)

    iteration = 0
    iter_per_test_epoch = n_test_samples/config['batch_size'] 
    for epoch in range(epochs):
        epoch_loss = 0.
        epoch_acc = 0.
        for iter_in_epoch in range(iter_per_epoch):
            epoch_x, epoch_fine, epoch_coarse = cifar100.next_train_batch(config['batch_size'])
            _, c, acc = sess.run([optimizer, loss, accuracy],#, summ], 
                            feed_dict = {mod.x: epoch_x, mod.y: epoch_coarse, 
                                         mod.training:True, cy_lr: Lr[iteration],
                                         mod.keep_prob: 0.7})
            epoch_loss += c
            epoch_acc += acc
            iteration+=1
            if iter_in_epoch%100 == 0:
                print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
                      'completed out of ', epochs, 'loss: ', epoch_loss/(iter_in_epoch+1),
                      'acc: ', '{:.2f}%'.format(epoch_acc*100/(iter_in_epoch+1)))
        print('######################')        
        print('TRAIN')        
        print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
              'completed out of ', epochs, 'loss: ', epoch_loss/int(iter_per_epoch),
              'acc: ', '{:.2f}%'.format(epoch_acc*100/int(iter_per_epoch)))
        train_loss_set.append(epoch_loss/int(iter_per_epoch))
        train_acc_set.append(epoch_acc*100/int(iter_per_epoch))
        test_loss = 0.
        test_acc = 0.
        for iter_in_epoch in range(int(iter_per_test_epoch)):            
            epoch_x, epoch_fine, epoch_coarse = cifar100.next_test_batch(config['batch_size'])
            c, acc = sess.run([loss, accuracy], 
                              feed_dict = {mod.x: epoch_x, mod.y: epoch_coarse, 
                                           mod.training:False, mod.keep_prob:1.})
            test_loss += c
            test_acc += acc
        print('TEST')        
        print('Epoch ', epoch,  'loss: ', test_loss/int(iter_per_test_epoch), 
              'acc: ', '{:.2f}%'.format(test_acc*100/int(iter_per_test_epoch)))
        print('###################### \n')     
        test_loss_set.append(test_loss/int(iter_per_test_epoch))
        test_acc_set.append(test_acc*100/int(iter_per_test_epoch))
        
    return train_loss_set, train_acc_set, test_loss_set, test_acc_set


def run_fine(ithx, ithy, test_ith, test_ithy, mod, pred, loss, epochs, base_lre, max_lr, name, var):
    train_loss_set = []
    train_acc_set = []
    test_loss_set = []
    test_acc_set = []
    iter_per_epoch = int(n_samples/(config['batch_size']*20))
    Lr = cycle_lr(base_lre, max_lr, iter_per_epoch, 
              config['cycle_epoch'], config['cycle_ratio'], epochs)

    cy_lr = tf.placeholder(tf.float32, shape=(),  name = "cy_lr_"+name)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cy_lr).minimize(loss, var_list = var)

    prediction = tf.argmax(pred, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(mod.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy_"+name)

    iteration = 0
#    iter_per_test_epoch = n_test_samples/config['batch_size'] 
    for epoch in range(epochs):
        epoch_loss = 0.
        epoch_acc = 0.
        for iter_in_epoch in range(iter_per_epoch):
            iter_idx = np.random.choice(range(2500), config['batch_size'] , replace= False)
            epoch_x, epoch_fine = ithx[iter_idx], ithy[iter_idx]
            _, c, acc = sess.run([optimizer, loss, accuracy],#, summ], 
                            feed_dict = {mod.x: epoch_x, mod.y: epoch_fine, 
                                         mod.training:True, cy_lr: Lr[iteration],
                                         mod.keep_prob: 0.7})
            epoch_loss += c
            epoch_acc += acc
            iteration+=1
        print('######################')        
        print('TRAIN')        
        print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
              'completed out of ', epochs, 'loss: ', epoch_loss/int(iter_per_epoch),
              'acc: ', '{:.2f}%'.format(epoch_acc*100/int(iter_per_epoch)))
        train_loss_set.append(epoch_loss/int(iter_per_epoch))
        train_acc_set.append(epoch_acc*100/int(iter_per_epoch))
        
        test_loss = 0.
        test_acc = 0.
        c, acc = sess.run([loss, accuracy], 
                          feed_dict = {mod.x: test_ith, mod.y: test_ithy, 
                                       mod.training:False, mod.keep_prob:1.})
        test_loss += c
        test_acc += acc
        print('TEST')        
        print('Epoch ', epoch,  'loss: ', test_loss, 
              'acc: ', '{:.2f}%'.format(test_acc*100))
        print('###################### \n')     
        test_loss_set.append(test_loss)
        test_acc_set.append(test_acc*100)
        
    return train_loss_set, train_acc_set, test_loss_set, test_acc_set


folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets'])
result_dic = {}

with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    

    ### run shared ###
    tr_l, tr_a, te_l, te_a = run(model, shared_pred, shared_loss, config['epochs_set'][0], 
                                 config['base_lr'][0], config['max_lr'][0], 's')
    result_dic['s'] =[tr_l, tr_a, te_l, te_a]

    
    ### run coarse ###
    coarse_vars =[]
    for layer in tf.trainable_variables():
        if layer.name[:6] in ['coarse']:
            coarse_vars.append(layer)
                 
    tr_l, tr_a, te_l, te_a = run_coarse(model_coarse, coarse_pred, coarse_loss, 
                                        config['epochs_set'][1], config['base_lr'][1], 
                                        config['max_lr'][1], 'c', var = coarse_vars)
    result_dic['c'] =[tr_l, tr_a, te_l, te_a]

    
    ### run fine ###
    for ith in range(config['num_coarse']):
        n = 'f{:02}'.format(ith)
        fine_vars = []
        for layer in tf.trainable_variables():
            if layer.name[:6] in ['fine{:02}'.format(ith)]:
                fine_vars.append(layer)        
        ith_idx = np.where(np.argmax(cifar100.train_labels_coarse,1)==ith)[0]
        test_ith_idx = np.where(np.argmax(cifar100.test_labels_coarse,1)==ith)[0]
        tr_l, tr_a, te_l, te_a = run_fine(cifar100.train_images[ith_idx], cifar100.train_labels[ith_idx],
                                          cifar100.test_images[test_ith_idx], cifar100.test_labels[test_ith_idx],
                                          model_fines[ith], fines_pred[ith], fines_loss[ith], 
                                          config['epochs_set'][2], config['base_lr'][1], 
                                          config['max_lr'][1], 'f{:02}'.format(ith), var = fine_vars)
        result_dic[n] =[tr_l, tr_a, te_l, te_a]

        
    ### final accuracy ###
    iter_per_epoch = int(n_samples/config['batch_size'])
    
    train_correction = 0
    for iter_in_epoch in range(iter_per_epoch):
        epoch_x, epoch_fine, epoch_coarse = cifar100.next_train_batch(config['batch_size'])
        coarse_info = sess.run(coarse_pred, 
                        feed_dict = {model_coarse.x: epoch_x, model_coarse.training:False, 
                                     model_coarse.keep_prob: 1.})
        pred_y = np.zeros(shape = (config['batch_size'], config['num_fines']))
        for ith in range(config['num_coarse']):
            fine_info = sess.run(fines_pred[ith], 
                            feed_dict = {model_fines[ith].x: epoch_x, 
                                         model_fines[ith].training:False, model_fines[ith].keep_prob: 1.})
            pred_y += np.multiply(coarse_info[:,ith].reshape(-1,1), fine_info)
        
        batch_prediction = np.argmax(pred_y,1)    
        train_correction += np.sum(np.equal(batch_prediction, np.argmax(epoch_fine,1))) 
   
    final_train_acc = train_correction/n_samples

    iter_test_per_epoch = int(n_test_samples/config['batch_size'])
    
    test_correction = 0
    for iter_in_epoch in range(iter_test_per_epoch):
        epoch_x, epoch_fine, epoch_coarse = cifar100.next_test_batch(config['batch_size'])
        coarse_info = sess.run(coarse_pred, 
                        feed_dict = {model_coarse.x: epoch_x, model_coarse.training:False, 
                                     model_coarse.keep_prob: 1.})
        pred_y = np.zeros(shape = (config['batch_size'], config['num_fines']))
        for ith in range(config['num_coarse']):
            fine_info = sess.run(fines_pred[ith], 
                            feed_dict = {model_fines[ith].x: epoch_x, 
                                         model_fines[ith].training:False, model_fines[ith].keep_prob: 1.})
            pred_y += np.multiply(coarse_info[:,ith].reshape(-1,1), fine_info)
        
        batch_prediction = np.argmax(pred_y,1)    
        test_correction += np.sum(np.equal(batch_prediction, np.argmax(epoch_fine,1))) 
   
    final_test_acc = test_correction/n_test_samples
    result_dic['final'] =[final_train_acc, final_test_acc] 
    print(result_dic['final'])





