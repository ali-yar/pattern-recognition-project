import glob
import numpy as np
from tflearn.data_utils import to_categorical
from sklearn import metrics
from makeData import getTestData
from makeModel import getModel

match = "models/*.tfl.index"

size_image = 16

results = []

for file in glob.glob(match):
    modelName = file[:-6]
    
    bnormid = modelName.find("bnorm"); bnormlen = len("bnorm")
    bsizeid = modelName.find("_bsize"); bsizelen = len("_bsize")
    lrateid = modelName.find("_lrate"); lratelen = len("_lrate")
    
    bNorm = modelName[bnormid + bnormlen : bsizeid]
    bSize = modelName[bsizeid + bsizelen : lrateid]
    lRate = modelName[lrateid + lratelen : modelName.find("_acc")]
    
    # get data
    X_Test, Y_Test = getTestData()
    
    # make one-hot format for labels
    Y_Test_cat = to_categorical(np.ravel(Y_Test),2)

    # build model
    model = getModel(size_image, lRate)
    
    # load model weights
    model.load(modelName)  
    
    # test model
    Y_pred = model.predict(X_Test)
    
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
    
results.append(" ")

# save results to file
with open("accuracyFullTest.txt","a") as f:
    f.write("\n********** Experiment Results **********\n")
    headers = "b_norm\tb_size\tl_rate\t\tacc\t\tauc\n"
    f.write(headers + "\n")
    for res in results:
        f.write(res + "\n")