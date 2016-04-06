import os, subprocess
import argparse
from argparse import ArgumentParser
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split

from Utilities import feval, writingfile, readfeats, getlabels

def split2(datafile,trfile,tfile,truefile):
    allfeat = readfeats(datafile)
    trfeat, tfeat, y_train, y_test = train_test_split(allfeat, getlabels(allfeat), test_size=float(1)/7, stratify=getlabels(allfeat))
    writingfile(trfile, trfeat)
    writingfile(tfile, tfeat)
    writingfile(truefile, y_test)

def split(datafile,trfile,vfile,tfile,vtruefile,truefile):
    allfeat = readfeats(datafile)
    afeat, tfeat, y_adapt, y_test = train_test_split(allfeat, getlabels(allfeat), test_size=float(1)/3, stratify=getlabels(allfeat))
    writingfile(tfile, tfeat)
    writingfile(truefile, y_test)

    trfeat, vfeat, y_train, y_validation = train_test_split(afeat, getlabels(afeat), test_size=float(3)/7, stratify=getlabels(afeat))
    writingfile(trfile, trfeat)
    writingfile(vfile, vfeat)
    writingfile(vtruefile, y_validation)

def predict(tfile,pfile,emo):
    model='../models/train.'+emo+'.adapted.model'
    predcmd=["../libSVM/svm-predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    p.communicate()

def adapt(ci,wi,s1,s2,s3,trfile,tfile,pfile,emo):
#    source_model1='../output/embedding/'+emo+'/train.'+emo+'.model,'+s1
#    source_model2='../output/lexicon/'+emo+'/train.'+emo+'.model,'+s2
#    source_model3='../output/ngrams/'+emo+'/train.'+emo+'.model,'+s3
    output_model='../models/train.'+emo+'.adapted.model'
    adaptcmd=["wine", "../aSVM/adapt_svm_train", "-c", "50", "-wi", "10", "-l", "1", "+", source_model1, "+", source_model2, "+", source_model3, trfile, output_model]
    adaptcmd[3]=ci
    adaptcmd[5]=wi
    adaptcmd[7]=1 #or 0 if manually weights are assigned
    p = subprocess.Popen(adaptcmd, stdout=subprocess.PIPE)
    p.communicate()
    predict(tfile,pfile,emo)

def CV(ci,wi,s1,s2,s3,trfile,CV_trfile,CV_tfile,CV_pfile,CV_truey,emo):
    feats = readfeats(trfile)
    cv = StratifiedShuffleSplit(y=getlabels(feats), n_iter=5, test_size=0.2, random_state=0)
    f1_list = []
    p_list = []
    r_list = []
    count = 0
    for train_index, test_index in cv:
        count+=1
        cv_trfile = CV_trfile+str(count)
        cv_tfile = CV_tfile+str(count)
        cv_pfile = CV_pfile+str(count)
        cv_truey = CV_truey+str(count)
        X_train=feats[train_index]
        X_test=feats[test_index]
        y_test = getlabels(X_test)
        writingfile(cv_trfile, X_train)
        writingfile(cv_tfile, X_test)
        writingfile(cv_truey, y_test)   
        adapt(str(ci),str(wi),str(s1),str(s2),str(s3),cv_trfile,cv_tfile,cv_pfile,emo)
        y_test, y_predicted = feval(cv_truey, cv_pfile)
        p_list.append(metrics.precision_score(y_test, y_predicted, average='binary'))
        r_list.append(metrics.recall_score(y_test, y_predicted, average='binary'))
        f1_list.append(metrics.f1_score(y_test, y_predicted, average='binary'))
    recall = np.mean(np.asarray(r_list))
    precision = np.mean(np.asarray(p_list))
    f1 = np.mean(np.asarray(f1_list))
    print "C=%s, wi=%s and (s1=%s ; s2=%s ; s3=%s), its F1 is %f"%(ci,wi,s1,s2,s3,f1)
    return recall, precision, f1

def tune(emo,trfile,vfile,pfile,truefile,CV_trfile,CV_tfile,CV_pfile,CV_truey):
    c = [50,100,0.1,1,10,30]
    weights = [0.1,0.3,0.5,0.8,1,3,5,8,10]
    s = [0]
#    s = [0.1,0.3,0.5,0.8,1,3,5,8,10]
    f1_list = []
    p_list = []
    r_list = []
    tune_=[]
    for ci in c:
        for wi in weights:
            for s1 in s:
                for s2 in s:
                    for s3 in s:
                        r, p, f1 = CV(ci,wi,s1,s2,s3,trfile,CV_trfile,CV_tfile,CV_pfile,CV_truey,emo)
                        tune_.append([ci,wi,s1,s2,s3,r,p,f1])
    tune_=sorted(tune_,key=lambda x: x[-1],reverse=True)
    return tune_


def main(data_dir):
    emotions = [x[1] for x in os.walk(data_dir)][0]
    data_split = "/30-30/" # define the test data split ratio here and make sure you have the sub folders
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
        cv_trfile = dir+'/cv/train.cv'
        cv_tfile = dir+'/cv/test.cv'
        cv_pfile = dir+'/cv/predresults.cv'
        cv_truey = dir+'/cv/y_test.cv'
        
        print "---Parameter tuning"
        tune_=tune(emo,trainfile,valfile,valpredfile,valtruefile,cv_trfile,cv_tfile,cv_pfile,cv_truey)
        bestc=tune_[0][0]
        bestw=tune_[0][1]
        bests1=tune_[0][2]
        bests2=tune_[0][3]
        bests3=tune_[0][4]
        bestr=tune_[0][-3]
        bestp=tune_[0][-2]
        bestf1=tune_[0][-1]
        print "Tuning aSVM on %s, the best F1 is %f at c=%s, wi=%s and (s1=%s ; s2=%s ; s3=%s)"%(valfile,bestf1,str(bestc),str(bestw),str(bests1),str(bests2),str(bests3))
        
        print "---Model fitting and prediction"
        adapt(str(bestc),str(bestw),str(bests1),str(bests2),str(bests3),trainfile,testfile,predfile,emo)
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
           

