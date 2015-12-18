"""
    This module is for cross-validating on Purver's data sets.
"""
import os, subprocess
import argparse
from argparse import ArgumentParser
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit

from Utilities import load_tweets, feval, frange, getlabels, writingfile, readfeats
from FeatureTransformer import feature_transformer

def scaling(dir):
    cmd1=["../libSVM/svm-scale", "-l", "-1", "-u", "1", "-s", "../liblinear/range", dir+"/training"]
    with open(dir+"/train.scale", "w") as outfile1:
        subprocess.call(cmd1, stdout=outfile1)
    cmd2=["../libSVM/svm-scale", "-r", "../liblinear/range", dir+"/testing"]
    with open(dir+"/test.scale", "w") as outfile2:
        subprocess.call(cmd2, stdout=outfile2)
    outfile1.close()
    outfile2.close()
    
def CV(ci,gamma,kernel,trfile,CV_trfile,CV_tfile,CV_pfile,CV_truey):
    d = readfeats(trfile)
    cv = ShuffleSplit(n=len(d), n_iter=5, test_size=0.2, random_state=0)
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
        X_train=d[train_index]
        X_test=d[test_index]
        y_test = getlabels(X_test)
        writingfile(cv_trfile, X_train)
        writingfile(cv_tfile, X_test)
        writingfile(cv_truey, y_test)   
        traincmd=["../libSVM/svm-train", "-c", "0.001", "-t", "2", "-q", cv_trfile]     
        traincmd[2]=ci
        traincmd[4]=kernel
        subprocess.call(traincmd)
        model=cv_trfile.split('/')[-1]+'.model'
        predcmd=["../libSVM/svm-predict", cv_tfile, model, cv_pfile]
        p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
        output, err = p.communicate()
        y_test, y_predicted = feval(cv_truey, cv_pfile)
        p_list.append(metrics.precision_score(y_test, y_predicted, average='binary'))
        r_list.append(metrics.recall_score(y_test, y_predicted, average='binary'))
        f1_list.append(metrics.f1_score(y_test, y_predicted, average='binary'))
    recall = np.mean(np.asarray(r_list))
    precision = np.mean(np.asarray(p_list))
    f1 = np.mean(np.asarray(f1_list))
    print "C=%s and gamma=(1/num_feature), its F1 is %f"%(ci,f1)
    return [recall, precision, f1]
            
def tuneC(trfile,tfile,cv_trfile,cv_tfile,cv_pfile,cv_truey):
    kernel = 2
    c = [10, 30, 50, 100, 1000, 0.1, 0.01, 0.05, 0.001, 0.0001]
#    c = [10]
    ga = float(1)/620 # though the default gamma value is used
    tunec=[]
    for ci in c:
#        for ga in gamma:
        cv_result=CV2(str(ci),str(ga),str(kernel),trfile,cv_trfile,cv_tfile,cv_pfile,cv_truey)
        f1=cv_result[-1]
        r=cv_result[0]
        p=cv_result[1]
        tunec.append([ci,ga,kernel,r,p,f1])
    tunec=sorted(tunec,key=lambda x: x[-1],reverse=True)
    return tunec
    
def main(ci,gamma,data_dir,scale,tune):
    emotions = [x[1] for x in os.walk(data_dir)][0]
    for emo in emotions:
        print 80*"*"
        print "Emotion is", emo
        dir = data_dir+emo
        trainfile = dir+"/train.scale"
        predfile = dir+"/pred"
        cv_trfile = dir+'/cv/train.cv'
        cv_tfile = dir+'/cv/test.cv'
        cv_pfile = dir+'/cv/predresults.cv'
        cv_truey = dir+'/cv/y_test.cv'
        if scale == 'True':
            print "---Feature scaling"
            scaling(dir)
        else:
            pass
        print '---Cross-validation'
        tunec=tuneC(trainfile,cv_trfile,cv_tfile,cv_pfile,cv_truey)
        bestc=tunec[0][0]
        bestF=tunec[0][-1]
        bestP=tunec[0][-2]
        bestR=tunec[0][-3]
#        for i in tunec:
#            print "CV: Learning LibSVM with kernel=rbf and C=%f and gamma=(1/num_feature), its F1 is %f"%(i[0],i[-1])
        print "Five-fold CV on %s, the best recall is %f; best precision is %f; best F1 is %f at c=%f"%(trainfile,bestR,bestP,bestF,bestc)
            
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--d", dest="d", help="folder name") # for testing
    parser.add_argument("--scaling", dest="sca", help="Feature scaling, default False", default='False')
    args = parser.parse_args()
    data_dir = '../output/'+args.d+'/'
    ci = 10
    gamma = 0.01  
    main(ci,gamma,data_dir,args.sca,args.tun)