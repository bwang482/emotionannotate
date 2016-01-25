import os
import subprocess
import argparse
from argparse import ArgumentParser
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split

from Utilities import feval, writingfile, readfeats, getlabels


def split2(datafile, trfile, tfile, truefile):
    allfeat = readfeats(datafile)
    trfeat, tfeat, y_train, y_test = train_test_split(allfeat, getlabels(allfeat), test_size=0.3, stratify=getlabels(allfeat))
    writingfile(trfile, trfeat)
    writingfile(tfile, tfeat)
    writingfile(truefile, y_test)


def split(datafile, trfile, vfile, tfile, vtruefile, truefile):
    allfeat = readfeats(datafile)
    afeat, tfeat, y_adapt, y_test = train_test_split(allfeat, getlabels(allfeat), test_size=float(1)/3, stratify=getlabels(allfeat))
    writingfile(tfile, tfeat)
    writingfile(truefile, y_test)

    trfeat, vfeat, y_train, y_validation = train_test_split(afeat, getlabels(afeat), test_size=float(3)/7, stratify=getlabels(afeat))
    writingfile(trfile, trfeat)
    writingfile(vfile, vfeat)
    writingfile(vtruefile, y_validation)


def predict(tfile, pfile, emo):
    model = '../models/all/train.'+emo+'.adapted.model'
    predcmd = ["../libSVM/svm-predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    p.communicate()


def adapt(ci, wi, si, trfile, tfile, pfile, emo):
#    source_model1='../output/embedding/'+emo+'/train.'+emo+'.model,'+s1
#    source_model2='../output/lexicon/'+emo+'/train.'+emo+'.model,'+s2
#    source_model3='../output/ngrams/'+emo+'/train.'+emo+'.model,'+s3
    output_model = '../models/all/train.'+emo+'.adapted.model'
#    adaptcmd = ["wine", "../aSVM/adapt_svm_train", "-c", "50", "-wi", "10", "+", source_model1, "+", source_model2, "+", source_model3, trfile, output_model]
    source_model = '../models/all/train.'+emo+'.aux.model,'+si
    adaptcmd = ["wine", "../aSVM/adapt_svm_train", "-c", "50", "-wi", "10", "+", source_model, trfile, output_model]
    adaptcmd[3] = ci
    adaptcmd[5] = wi
#    adaptcmd[9] = si
    p = subprocess.Popen(adaptcmd, stdout=subprocess.PIPE)
    p.communicate()
    predict(tfile, pfile, emo)


def tune(emo, trfile, vfile, pfile, truefile):
    c = [50,100,0.1,1,10,30,0.01]
    weights = [0.1,0.3,0.5,0.8,1,3,5,8,10]
    s = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10]
    f1_list = []
    p_list = []
    r_list = []
    tune_ = []
    si = [1]
    for ci in c:
        for wi in weights:
            for si in s:
                adapt(str(ci), str(wi), str(si), trfile, vfile, pfile, emo)
                y_test, y_predicted = feval(truefile, pfile)
                f1 = metrics.f1_score(y_test, y_predicted, average='binary')
                r = metrics.recall_score(y_test, y_predicted, average='binary')
                p = metrics.precision_score(y_test, y_predicted, average='binary')
                tune_.append([ci, wi, si, r, p, f1])
                print "C=%s, wi=%s and s=%s, its F1 is %f"%(ci,wi,si,f1)
    tune_ = sorted(tune_, key=lambda x: x[-1], reverse=True)
    return tune_


def main(data_dir):
    emotions = [x[1] for x in os.walk(data_dir)][0]
    data_split = "/70-30/"  # define the test data split ratio here and make sure you have the sub folders
    for emo in emotions:
        print 80*"*"
        print "Emotion is", emo
        dir = data_dir+emo
        datafile = dir+"/test.scale"
#        predfile = dir+"/pred"
#        truefile = dir+'/y_test'
        trainfile = dir+data_split+emo+".train"
        valfile = dir+data_split+emo+".val"
        valpredfile = dir+data_split+emo+".val.pred"
        valtruefile = dir+data_split+emo+".val.true"
        testfile = dir+data_split+emo+".test"
        predfile = dir+data_split+emo+".test.pred"
        truefile = dir+data_split+emo+".test.true"
#        split2(datafile,trainfile,testfile,truefile)

        print "---Parameter tuning"
        tune_ = tune(emo, trainfile, testfile, predfile, truefile)
        bestc = tune_[0][0]
        bestw = tune_[0][1]
        bests = tune_[0][2]
        bestr = tune_[0][-3]
        bestp = tune_[0][-2]
        bestf1 = tune_[0][-1]
        print "Tuning aSVM on %s, the best F1 is %f at c=%s, wi=%s and s=%s"%(valfile,bestf1,str(bestc),str(bestw),str(bests))

        print "---Model fitting and prediction"
#        adapt(str(30),str(0.1),str(5),adaptfile,testfile,predfile,emo)
        adapt(str(bestc), str(bestw), str(bests), trainfile, testfile, predfile, emo)
        print "---Evaluation"
        y_test, y_predicted = feval(truefile, predfile)
        print 'Precision score: ', metrics.precision_score(y_test, y_predicted, average='binary')
        print 'Recall score: ', metrics.recall_score(y_test, y_predicted, average='binary')
        print 'F1 score: ', metrics.f1_score(y_test, y_predicted, average='binary')
        print 80*"*"


if __name__ == "__main__":
    # Please make sure there are 5 sub folders for each emotion within the output folder e.g. test1, provided below:
    data_dir = '../output/test1/'
    main(data_dir)
