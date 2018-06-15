import os
import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.externals import joblib

# Generate the data once before running the experiments. This is to ensure
# the same data is being used over all the experiments
def generateData(totalTrainPos, totalTrainNeg, totalTestPos, totalTestNeg):
    createTrainData(totalTrainPos, totalTrainNeg)
    createTestData(totalTestPos, totalTestNeg)


def createTrainData(totalPos, totalNeg):
    data_dir = "data/"
    
    # POSITIVE - TRAIN
    fpos = h5py.File(data_dir + 'Positive_train.h5', 'r')
    a=list(fpos.keys())[0]
    dpos=fpos[a]
    size_image=dpos.shape[1]
    num_pos_train = dpos.shape[0]
    dposar = np.zeros(dpos.shape)
    dpos.read_direct(dposar)
    train_pos=dposar.reshape([-1,size_image,size_image,1])
    
    print("The positive patches for training are: ", num_pos_train)
    
    # NEGATIVE - TRAIN
    fneg = h5py.File(data_dir + '300kNegative_train.h5', 'r')
    a=list(fneg.keys())[0]
    dneg=fneg[a]
    size_image=dneg.shape[1]
    num_neg_train = dneg.shape[0]
    dnegar = np.zeros(dneg.shape)
    dneg.read_direct(dnegar)
    train_neg=dnegar.reshape([-1,size_image,size_image,1])
    
    print("The negative patches for training are: ", num_neg_train)
    
    actual_num_pos_train = totalPos
    actual_num_neg_train = totalNeg
    
    # Create the target vectors
    train_pos_lab = np.ones((actual_num_pos_train,1))
    train_neg_lab = np.zeros((actual_num_neg_train,1))
    
    #### Build the training set  (images and targets)
    tpos = np.copy(train_pos[0:actual_num_pos_train,:,:,:])
    tneg = np.copy(train_neg[0:actual_num_neg_train,:,:,:])
    
    # Stack the subsets
    X_Train = np.vstack((tpos,tneg))
    Y_Train = np.vstack((train_pos_lab,train_neg_lab))
    
    # Shuffle the two arrays in unison
    X_Train, Y_Train = shuffle(X_Train,Y_Train)
    
    # Save
    joblib.dump(X_Train, data_dir + "X_Train.pkl")
    joblib.dump(Y_Train, data_dir + "Y_Train.pkl")


def createTestData(totalPos, totalNeg):
    data_dir = "data/"
    	
    # POSITIVE - Test
    fpos = h5py.File(data_dir + 'Positive_test.h5', 'r')
    a=list(fpos.keys())[0]
    dpos=fpos[a]
    size_image=dpos.shape[1]
    num_pos_test = dpos.shape[0]
    dposar = np.zeros(dpos.shape)
    dpos.read_direct(dposar)
    test_pos=dposar.reshape([-1,size_image,size_image,1])
    
    print("The positive patches for testing are: ", num_pos_test)
    
    # NEGATIVE - Test
    fneg = h5py.File(data_dir + '300kNegative_test.h5', 'r')
    a=list(fneg.keys())[0]
    dneg=fneg[a]
    size_image=dneg.shape[1]
    num_neg_test = dneg.shape[0]
    dnegar = np.zeros(dneg.shape)
    dneg.read_direct(dnegar)
    test_neg=dnegar.reshape([-1,size_image,size_image,1])
    
    print("The negative patches for testing are: ", num_neg_test)
    
    actual_num_pos_test = totalPos
    actual_num_neg_test = totalNeg
    
    # Create the target vectors
    test_pos_lab = np.ones((actual_num_pos_test,1))
    test_neg_lab = np.zeros((actual_num_neg_test,1))
    
    #### Build the testing set  (images and targets)
    tpos = np.copy(test_pos[0:actual_num_pos_test,:,:,:])
    tneg = np.copy(test_neg[0:actual_num_neg_test,:,:,:])
    
    # Stack the subsets
    X_Test = np.vstack((tpos,tneg))
    Y_Test = np.vstack((test_pos_lab,test_neg_lab))
    
    # Save
    joblib.dump(X_Test, data_dir + "X_Test.pkl")
    joblib.dump(Y_Test, data_dir + "Y_Test.pkl")
    
    
def getData():
    data_dir = "data/"
    X_Train = joblib.load(data_dir + "X_Train.pkl")
    Y_Train = joblib.load(data_dir + "Y_Train.pkl")
    X_Test = joblib.load(data_dir + "X_Test.pkl")
    Y_Test = joblib.load(data_dir + "Y_Test.pkl")
    
    return X_Train, Y_Train, X_Test, Y_Test
