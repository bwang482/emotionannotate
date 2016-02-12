import os
import subprocess
import json
import numpy as np
from sklearn import metrics
import multiprocessing as mp

from Utilities import feval, writingfile, readfeats
from FeatureTransformer import feature_transformer2, check_csr


def scaling(dir, emo):
    cmd2 = ["../libSVM/svm-scale", "-r", "../libSVM/"+emo+".range", dir+".testing"]
    with open(dir+".test.scale", "w") as outfile2:
        subprocess.call(cmd2, stdout=outfile2)
    outfile2.close()


def writevec(filename, x, y):
    f = open(filename, 'wb')
    for i in xrange(x.shape[0]):
        f.write(str(y)+'\t')
        feature = x[i].toarray()
        for (j, k) in enumerate(feature[0]):
            f.write(str(j+1)+':'+str(k)+' ')
        f.write('\n')
    f.close()


def predict(tfile, pfile, emo):
    model = '../models/'+'train.'+emo+'.adapted.model'
    predcmd = ["../libSVM/svm-predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    p.communicate()


def classifier(data, emo, output, lexicon_feat, embed_feat):
    preds = []
    pred = {}
    dir = '../output/features/'+emo
    testfile = dir+".test.scale"
    predfile = dir+".pred"
#    truefile = dir+'.true'
    feat = feature_transformer2(data, emo, lexicon_feat, embed_feat)
    feat = check_csr(feat)
    writevec(dir+'.testing', feat, 1)
    scaling(dir, emo)
    predict(testfile, predfile, emo)
    with open(predfile, "r") as f:
        for l in f.readlines():
            l = l.strip()
            if (l == '1') or (l == 1):
                p = 'yes'
            else:
                p = 'no'
            preds.append(p)
    pred[emo] = preds
    output.put(pred)

def sequentClassifier(input, lexicon_feat, embed_feat):
    pred = {}
    preds = []
    emotions = ['anger', 'disgust', 'happy', 'sad', 'surprise']
    for emo in emotions:
        temp_pred = []
        dir = '../output/features/'+emo
        testfile = dir+".test.scale"
        predfile = dir+".pred"
        feat = feature_transformer2(input, emo, lexicon_feat, embed_feat)
        feat = check_csr(feat)
        writevec(dir+'.testing', feat, 1)
        scaling(dir, emo)
        predict(testfile, predfile, emo)
        with open(predfile, "r") as f:
            for l in f.readlines():
                l = l.strip()
                if (l == '1') or (l == 1):
                    p = 'yes'
                else:
                    p = 'no'
                temp_pred.append(p)
        pred[emo] = temp_pred
        
    for i in xrange(len(input)):
        temp_pred = {}
        if 'text' in input[i].keys():
            temp_pred['text'] = input[i]['text']
        if 'tweetid' in input[i].keys():
            temp_pred['tweetid'] = input[i]['tweetid']
        emo = {}
        for e in emotions:
            emo[e] = pred[e][i]
        temp_pred['emotions'] = emo
        preds.append(temp_pred)
    return preds  


def parallelClassifier(input, lexicon_feat, embed_feat):
    preds = []
    output = mp.Queue()
    emotions = ['anger', 'disgust', 'happy', 'sad', 'surprise']
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
        pred = {}
        if 'text' in input[i].keys():
            pred['text'] = input[i]['text']
        if 'tweetid' in input[i].keys():
            pred['tweetid'] = input[i]['tweetid']
        emo = {}
        for e in emotions:
            emo[e] = results[e][i]
        pred['emotions'] = emo
        preds.append(pred)
    return preds


def initFeatureProcessors():
    from LexiconFeature import LexiconVectorizer
    from EmbeddingFeature import EmbeddingVectorizer
    lexicon_feat = LexiconVectorizer()
    embed_feat = EmbeddingVectorizer()
    return lexicon_feat, embed_feat
