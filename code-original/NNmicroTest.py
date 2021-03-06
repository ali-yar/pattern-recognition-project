
from __future__ import division, print_function, absolute_import


import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn import metrics
import h5py

#################################################################################################################

# POSITIVE - Test
fpos = h5py.File('Positive_test.h5', 'r')
a=list(fpos.keys())[0]
dpos=fpos[a]
size_image=dpos.shape[1]
num_pos_test = dpos.shape[0]
dposar = np.zeros(dpos.shape)
dpos.read_direct(dposar)
test_pos=dposar.reshape([-1,size_image,size_image,1])

print("The positive patches for testing are: ", num_pos_test)

# NEGATIVE - Test
fneg = h5py.File('300kNegative_test.h5', 'r')
a=list(fneg.keys())[0]
dneg=fneg[a]
size_image=dneg.shape[1]
num_neg_test = dneg.shape[0]
dnegar = np.zeros(dneg.shape)
dneg.read_direct(dnegar)
test_neg=dnegar.reshape([-1,size_image,size_image,1])

print("The negative patches for testing are: ", num_neg_test)

actual_num_pos_test = int(input("Enter the number of positive samples to be used for testing: "))
actual_num_neg_test = int(input("Enter the number of negative samples to be used for testing: "))

# Create the target vectors
test_pos_lab = np.ones((actual_num_pos_test,1))
test_neg_lab = np.zeros((actual_num_neg_test,1))

#### Build the testing set  (images and targets)
tpos = np.copy(test_pos[0:actual_num_pos_test,:,:,:])
tneg = np.copy(test_neg[0:actual_num_neg_test,:,:,:])

# Stack the subsets
X_Test = np.vstack((tpos,tneg))
Y_Test = np.vstack((test_pos_lab,test_neg_lab))


#################################################################################################################



with tf.Graph().as_default():

    ###################################
    # Image transformations
    ###################################
    
    # normalisation of images
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    # Create extra synthetic training data by flipping & rotating images
    print ("----Begin data augmentation ----------")
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    print ("----End data augmentation ----------")
    
    ####################################################################################################
    
    ################################### Define NETWORK ARCHITECTURE ###################################
    
    ###################################################################################################
    
    
    
    # Input is a 16x16 image
    input_layer = input_data(shape=[None, size_image, size_image, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    
    # 1: Convolution layer with 32 filters, each 3x3x3
    conv_1 = conv_2d(input_layer, 32, 3, activation='relu', name='conv_1')
    
    # 2: Convolution layer with 32 filters
    conv_2 = conv_2d(conv_1, 32, 3, activation='relu', name='conv_2')
    
    # 3: Max pooling layer
    mp1 = max_pool_2d(conv_2, 2)
    
    # 4: Convolution layer with 32 filters
    conv_3 = conv_2d(mp1, 32, 3, activation='relu', name='conv_3')
     
    # 5: Convolution layer with 32 filters
    conv_4 = conv_2d(conv_3, 32, 3, activation='relu', name='conv_4')
     
    # 5: Max pooling layer
    mp2 = max_pool_2d(conv_4, 2)
     
    # 6: Fully-connected 256 node layer
    dense1 = fully_connected(mp2, 256, activation='relu')
     
    # 7: Dropout layer to combat overfitting
    dp1 = dropout(dense1, 0.5)
    
    dense2 = fully_connected(dp1, 256, activation='relu')
    
    dp2 = dropout(dense2, 0.5)
    
    
    
    # 8: Fully-connected layer with 1 outputs ( binary)
    network = fully_connected(dp2, 2, activation='softmax')   
        
    # Configure how the network will be trained
    
    gd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=500)
    
    network = regression(network, optimizer=gd, loss='categorical_crossentropy')
    
    # Wrap the network in a model object
    
    model = tflearn.DNN(network, tensorboard_verbose=0)
    
    Y = to_categorical(np.ravel(Y_Test),2)
    
    print(" -------------------- Load Module --------------------------------------")    
    # Save the model
    model.load('model/model_calc_tiss_1_final.tfl')
    
    Y_pred = model.predict(X_Test)
    
    
    fpr, tpr, thresholds = metrics.roc_curve(Y, Y_pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    
    print("AUC: {}%".format(auc))
    

