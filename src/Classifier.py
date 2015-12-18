import os, subprocess, json
import numpy as np
from sklearn import metrics
import multiprocessing as mp

from Utilities import feval, writingfile, readfeats
from FeatureTransformer import feature_transformer2, check_csr

def scaling(dir):
    cmd2=["../libSVM/svm-scale", "-r", "../liblinear/range", dir+"/testing"]
    with open(dir+"/test.scale", "w") as outfile2:
        subprocess.call(cmd2, stdout=outfile2)
    outfile2.close()

def writevec(filename,x,y):
    f=open(filename,'wb')
#    for i in xrange(len(y)):
    f.write(str(y)+'\t')
    feature=x.toarray()
    for (j,k) in enumerate(feature[0]):
        f.write(str(j+1)+':'+str(k)+' ')
#    f.write('\n')
    f.close() 

def predict(tfile,pfile,emo):
    model='../models/'+'train.'+emo+'.model'
    predcmd=["../libSVM/svm-predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    p.communicate()

#def classifier(data):
#    output = {}
#    output['anger'] = 'test!'
#    output['disgust'] = 'test!'
#    output['happy'] = 'test!'
#    output['sad'] = 'test!'
#    output['surprise'] = 'test!'
#    return output

def classifier(data,emo,output):
    preds={}
    dir = '../output/temp2/'+emo
    testfile = dir+"/test.scale"
    predfile = dir+"/pred"
    truefile = dir+'/y_test'
    feat = feature_transformer2(data,emo)
    feat = check_csr(feat)
    writevec(dir+'/testing',feat,1)
    scaling(dir)
    predict(testfile, predfile,emo)
    f2 = open(predfile, 'r')
    l2 = f2.readlines()[0].strip()
    if (l2 == '1') or (l2 == 1):
        p='yes'
    else:
        p='no'
    preds[emo] = p
    output.put(preds)
        
def parallelClassifier(input):
    preds={}
    if 'text' in input.keys():
        preds['text'] = input['text']
    if 'tweetid' in input.keys():
        preds['tweetid'] = input['tweetid']
    output = mp.Queue()
    emotions = ['anger','disgust','happy','sad','surprise']
    processes = [mp.Process(target=classifier, args=(input, emo, output)) for emo in emotions]
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # Get process results from the output queue
    results_list = [output.get() for p in processes]
    results = {}
    for item in results_list:
       results[item.keys()[0]] = item.values()[0]
    preds['emotions'] = results
    return preds

if __name__ == "__main__":
    input = json.dumps({'tweetid': '111111', 'text': 'lucky @USERID ! good luck @USERID & see you soon :) @USERID @USERID'})
    input = json.loads(input)
    parallelClassifier(input)
    