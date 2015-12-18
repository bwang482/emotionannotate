from sklearn.base import BaseEstimator
import re, os
import numpy as np
import pandas as pd
import gensim

class streamdata(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.rstrip().split('\t')
                
def readTang(fname='../word2vec/sswe'):
    embs=streamdata(fname)
    embedmodel={}
    for tw2vec in embs:
        wd=tw2vec[0]
        value=[float(i) for i in tw2vec[1:]]
        embedmodel[wd]=np.array(value)
    return embedmodel

class EmbeddingVectorizer(BaseEstimator):
    def __init__(self,w2vf='../word2vec/w2v/c10_w3_s100',sswef='../word2vec/sswe'):#,adam='../word2vec/adam/adam_embeddings.csv'):
        self.w2v=gensim.models.Word2Vec.load(w2vf) 
        self.sswe=readTang(sswef) 
#        self.adam=pd.read_csv('../word2vec/adam/adam_embeddings.csv', index_col=0, sep=',',quotechar=' ')
    def emdsswe(self,uni):
        f=np.array([])    
        f=self.sswe.get(uni,self.sswe['<unk>'])
        return f
    def emdw2v(self,uni):
        f=np.array([])    
        try:
            f=self.w2v[uni]
        except:
            pass   
        return f
    def emdadam(self,uni):
        f=np.array([])
        if not uni.isdigit():
            try:
                f=adam.ix[uni].ravel()
            except:
                pass
        return f
    def concattw(self,feature,size,tw,etype):
        feat=np.array([]) 
        for i,uni in enumerate(tw):
            if etype=='w2v':
                f=self.emdw2v(uni)
            if etype=='sswe':
                f=self.emdsswe(uni)
            if etype=='adam':
                f=self.emdadam(uni)
            feat=np.concatenate([feat,f])
        if list(feat)==[]:
            feat=np.zeros((2*size,)) 
        if len(feat)<=size:
            feat=np.concatenate([feat,np.zeros((size,))])
        feat=feat.reshape(len(feat)/size,size)
        feature=np.concatenate([feature,feat.sum(axis=0)])
        feature=np.concatenate([feature,feat.max(axis=0)])
#        feature=np.concatenate([feature,feat.min(axis=0)])
        feature=np.concatenate([feature,feat.mean(axis=0)]) 
#        feature=np.concatenate([feature,feat.std(axis=0)])
#        feature=np.concatenate([feature,feat.prod(axis=0)])
        return feature   
    def fit(self, documents, y=None):
        return self              
    def transform(self, documents):
        x=np.array([])
        size1=len(self.w2v['the'])
        size2=len(self.sswe['the'])
#        size3=len(self.adam.ix['the'])
        for tweet in documents:
            d = tweet.lower()
            tw = d.split()
            feature=np.array([])
            feature=self.concattw(feature,size1,tw,'w2v') 
            feature=self.concattw(feature,size2,tw,'sswe')
#            feature=self.concattw(feature,size3,tw,'adam')
            x=np.concatenate([x,feature])
        x=x.reshape((len(documents),len(x)/len(documents)))
        return x






