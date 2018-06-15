from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
import scikitplot as skplt
import matplotlib.pyplot as plt
from tflearn.data_utils import to_categorical
from sklearn import metrics
from makeData import getData
from makeModel import getModel


################################ Set options ##################################

generateData_flag = False
batch_size = [64, 128]
batch_norm = [False] # for now, not implemented
lRatePowers = np.array([4, 3]) # positive integer because.. \|/
learning_rate = 1/pow(2,lRatePowers) #              .. 1/2^p = 2^(-p)

epochs = 100
size_image = 16

################################ Generate data ################################

###############################################################################
# With 1 run of this script, all experiments (with different combination of 
# params set above) will be performed. If need to stop the execution of this
# script and later restart it (e.g. with different params), then we need to
# ensure the same training and test sets are being used. Therefore the below
# lines should be executed to generate those data only the first time by 
# setting the variable above "generateData_flag" to True and then turned to
# False before performing future runs of this script.
###############################################################################

if (generateData_flag) :
    from makeData import generateData
    totalTrainPos = 1500      # max = 3459
    totalTrainNeg = 3000     # max = 304225
    
    totalTestPos = 500     # max = 1164
    totalTestNeg = 500      # max = 110000
    
    generateData(totalTrainPos, totalTrainNeg, totalTestPos, totalTestNeg)

###############################################################################

# create folder to save ROC plots
plots_dir = "plots/"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
# create folder to save models
models_dir = "models/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
        
headers = "b_norm   l_rate  b_size   acc    auc\n"
print(headers)

results = []

for bNorm in batch_norm:
    for lRate in learning_rate:
        for bSize in batch_size:  
            
            # get data
            X_Train, Y_Train, X_Test, Y_Test = getData()
            
            # make one-hot format for labels
            Y_Train_cat = to_categorical(np.ravel(Y_Train),2)
            Y_Test_cat = to_categorical(np.ravel(Y_Test),2)
        
            # build model
            model = getModel(size_image, lRate)
            
            # train model
            model.fit(X_Train, Y_Train_cat, batch_size=bSize, n_epoch=epochs,
                                              show_metric=True, shuffle=True)    
            
            # test model
            Y_pred = model.predict(X_Test)
            
            # compute accuracy
            acc = metrics.accuracy_score(np.ravel(Y_Test),np.argmax(Y_pred,axis=1))
            acc = np.round(acc,4) * 100
            
            # compute AUC
            fpr, tpr, _ = metrics.roc_curve(Y_Test_cat.ravel(), Y_pred.ravel())
            auc = metrics.auc(fpr, tpr)
            auc = np.round(auc,4) * 100
            
            res = "{} \t {} \t {} \t {} \t {}".format(bNorm,lRate,bSize,acc,auc)
            results.append(res)
            print(res)
            
            # ROC curve
            skplt.metrics.plot_roc(Y_Test, Y_pred)
            plotName = "plot-bnorm{}_lrate{}_bsize{}_acc{}_auc{}.jpg".format(
                                                    bNorm,lRate,bSize,acc,auc)
            plt.savefig(plots_dir + plotName)
            
            # Save the model
            modelName = "model-bnorm{}_lrate{}_bsize{}_acc{}_auc{}.tfl".format(
                                                    bNorm,lRate,bSize,acc,auc)
            model.save(models_dir + modelName)
    
            # reset tensorflow graph            
            tf.reset_default_graph()

# save results to file
with open("accuracy.txt","a") as f:
    f.write("\n********** Experiment Results **********\n")
    f.write(headers + "\n")
    for res in results:
        f.write(res + "\n")

