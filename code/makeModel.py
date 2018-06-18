
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def getModel(size_image, is_dropout, learning_rate, is_batch_norm):
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
    
    ###########################################################################
    
    ################################### Define NETWORK ARCHITECTURE ###########
    
    ###########################################################################
    
#    winit = tflearn.initializations.uniform(minval=0, maxval=None, seed=0)

    
    # Input is a 16x16 image
    input_layer = input_data(shape=[None, size_image, size_image, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    
    # 1: Convolution layer with 32 filters, each 3x3x3
    conv_1 = conv_2d(input_layer, 32, 3, activation='relu', name='conv_1' )
    
    # 2: Convolution layer with 32 filters
    conv_2 = conv_2d(conv_1, 32, 3, activation='relu', name='conv_2' )
    
    # 3: Max pooling layer
    mp1 = max_pool_2d(conv_2, 2)
    
    # Batch normalization
    if is_batch_norm : mp1 = tflearn.normalization.batch_normalization(mp1)
    
    # 4: Convolution layer with 32 filters
    conv_3 = conv_2d(mp1, 32, 3, activation='relu', name='conv_3' )
     
    # 5: Convolution layer with 32 filters
    conv_4 = conv_2d(conv_3, 32, 3, activation='relu', name='conv_4' )
     
    # 5: Max pooling layer
    mp2 = max_pool_2d(conv_4, 2)
    
    # Batch normalization
    if is_batch_norm : mp2 = tflearn.normalization.batch_normalization(mp2)
     
    # 6: Fully-connected 256 node layer
    dense1 = fully_connected(mp2, 256, activation='relu' )
     
    # 7: Dropout layer to combat overfitting
    dp1 = dropout(dense1, 0.5) if is_dropout else dense1
    
    dense2 = fully_connected(dp1, 256, activation='relu' )
    
    dp2 = dropout(dense2, 0.5) if is_dropout else dense2
    
    # 8: Fully-connected layer with 1 outputs ( binary)
    network = fully_connected(dp2, 2, activation='softmax' )
        
    # Configure how the network will be trained
    
    gd = tflearn.SGD(learning_rate=learning_rate, lr_decay=0.96, decay_step=500)
    
    network = regression(network, optimizer=gd, loss='categorical_crossentropy', shuffle_batches=True)
    
    # Wrap the network in a model object
    
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model