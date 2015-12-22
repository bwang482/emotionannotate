import os, subprocess, json
import numpy as np
from sklearn import metrics
import multiprocessing as mp

from Utilities import feval, writingfile, readfeats
from FeatureTransformer import feature_transformer2, check_csr

def scaling(dir,emo):
    cmd2=["../libSVM/svm-scale", "-r", "../libSVM/"+emo+".range", dir+"/testing"]
    with open(dir+"/test.scale", "w") as outfile2:
        subprocess.call(cmd2, stdout=outfile2)
    outfile2.close()
    
def writevec(filename,x,y):
    f=open(filename,'wb')
    for i in xrange(x.shape[0]):
        f.write(str(y)+'\t')
        feature=x[i].toarray()
        for (j,k) in enumerate(feature[0]):
            f.write(str(j+1)+':'+str(k)+' ')
        f.write('\n')
    f.close() 

def predict(tfile,pfile,emo):
    model='../models/'+'train.'+emo+'.model'
    predcmd=["../libSVM/svm-predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    p.communicate()

def classifier(data,emo,output, lexicon_feat, embed_feat):
    preds=[]
    pred={}
    dir = '../output/temp2/'+emo
    testfile = dir+"/test.scale"
    predfile = dir+"/pred"
    truefile = dir+'/y_test'
    feat = feature_transformer2(data,emo, lexicon_feat, embed_feat)
    feat = check_csr(feat)
    writevec(dir+'/testing',feat,1)
    scaling(dir,emo)
    predict(testfile, predfile,emo)
    with open(predfile, "r") as f:
        for l in f.readlines():
            l = l.strip()
            if (l == '1') or (l == 1):
                p='yes'
            else:
                p='no'
            preds.append(p)
    pred[emo]=preds
    output.put(pred)
        
def parallelClassifier(input, lexicon_feat, embed_feat):
    preds=[]
    output = mp.Queue()
    emotions = ['anger','disgust','happy','sad','surprise']
    processes = [mp.Process(target=classifier, args=(input, emo, output, lexicon_feat, embed_feat)) for emo in emotions]
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
    for i in xrange(len(input)):
        pred={}
        if 'text' in input[i].keys():
            pred['text'] = input[i]['text']
        if 'tweetid' in input[i].keys():
            pred['tweetid'] = input[i]['tweetid']
        emo={}
        for e in emotions:
            emo[e] = results[e][i]
        pred['emotions']=emo
        preds.append(pred)
    return preds

def initFeatureProcessors():
    from LexiconFeature import LexiconVectorizer
    from EmbeddingFeature import EmbeddingVectorizer
    lexicon_feat = LexiconVectorizer()
    embed_feat = EmbeddingVectorizer()
    return lexicon_feat, embed_feat

if __name__ == "__main__":
    # For testing purpose:
    import datetime
    print "Start\t"+str(datetime.datetime.now())
    lexicon_feat, embed_feat = initFeatureProcessors()
    print "Initialised\t"+str(datetime.datetime.now())
    print "Classifying 1st tweet..."
    input = json.dumps({'tweetid': '111111', 'text': 'lucky @USERID ! good luck @USERID & see you soon :) @USERID @USERID'})
    input = json.loads(input)
    parallelClassifier([input], lexicon_feat, embed_feat)
    print "Done\t"+str(datetime.datetime.now())
    print "Classifying second tweet..."
    input = json.dumps({'tweetid': '111112', 'text': 'such an AWFUL experience!'})
    input = json.loads(input)
    parallelClassifier([input], lexicon_feat, embed_feat)
    print "Done\t"+str(datetime.datetime.now())
