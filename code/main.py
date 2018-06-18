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
batch_norm = [False] # for now, not implemented
batch_size = np.array([2, 16, 128, 1024])
learning_rate = np.array([ [0.0005, 0.0020, 0.0078, 0.0313], 
                           [0.0039, 0.0156, 0.0625, 0.2500], 
                           [0.0313, 0.1250, 0.5000, 2.0000], 
                           [0.2500, 1.0000, 4.0000, 16.000] ])

batch_size = np.array([16])
learning_rate = np.array([ [0.0005, 0.0020] ])

epochs = 1
dropout = False
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

        
results = []

for bNorm in batch_norm:         
    for i, bSize in enumerate(batch_size):
        for lRate in learning_rate[i]:
            
            # get data
            X_Train, Y_Train, X_Test, Y_Test = getData()
            
            # make one-hot format for labels
            Y_Train_cat = to_categorical(np.ravel(Y_Train),2)
            Y_Test_cat = to_categorical(np.ravel(Y_Test),2)
        
            # build model
            model = getModel(size_image, dropout, lRate, bNorm)
            
            # train model
            model.fit(X_Train, Y_Train_cat, batch_size=bSize, n_epoch=epochs,
                                              show_metric=True, shuffle=True)    
            
            # test model
            Y_pred = np.array([])
            splits = np.split(X_Test,4)
            for s in range(4):
                pred = model.predict(splits[s])
                if Y_pred.size == 0 :
                    Y_pred = pred
                else:
                    Y_pred = np.concatenate((Y_pred, pred),axis=0)
            
            # compute accuracy
            acc = metrics.accuracy_score(np.ravel(Y_Test),np.argmax(Y_pred,axis=1))
            acc = np.round(acc*100,2)
            
            # compute AUC
            fpr, tpr, _ = metrics.roc_curve(Y_Test_cat.ravel(), Y_pred.ravel())
            auc = metrics.auc(fpr, tpr)
            auc = np.round(auc*100,2)
            
            res = "{}\t{}\t{}\t\t{}\t\t{}"
            res = res.format(bNorm,bSize,lRate,acc,auc)
            
            results.append(res)
            print(res)
            
            # ROC curve
#            skplt.metrics.plot_roc(Y_Test, Y_pred)
#            
#            plotName = "plot-bnorm{}_lrate{}_bsize{}_acc{}_auc{}.jpg"
#            plotName = plotName.format(bNorm,bSize,lRate,acc,auc)
#            
#            plt.savefig(plots_dir + plotName)
            
            # Save the model
            modelName = "model-bnorm{}_bsize{}_lrate{}_acc{}_auc{}.tfl"
            modelName = modelName.format(bNorm,bSize,lRate,acc,auc)

            model.save(models_dir + modelName)
    
            # reset tensorflow graph            
            tf.reset_default_graph()
        results.append("_____________________________________________________")
        results.append(" ")

# save results to file
with open("accuracy.txt","a") as f:
    f.write("\n********** Experiment Results **********\n")
    headers = "b_norm\tb_size\tl_rate\t\tacc\t\tauc\n"
    f.write(headers + "\n")
    for res in results:
        f.write(res + "\n")
