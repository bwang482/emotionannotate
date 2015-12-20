import numpy as np
import argparse
import scipy.sparse
from argparse import ArgumentParser
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import svm
#from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel

import Config
from Utilities import load_tweets, format_data
from Preprocessor import preprocess
from LexiconFeature import LexiconVectorizer
from EmbeddingFeature import EmbeddingVectorizer

def writevec(filename,x,y):
    f=open(filename,'wb')
    for i in xrange(len(y)):
        f.write(str(y[i])+'\t')
        feature=x[i].toarray()
        for (j,k) in enumerate(feature[0]):
            f.write(str(j+1)+':'+str(k)+' ')
        f.write('\n')
    f.close() 
    
def writey(filename, y):
    f=open(filename,'wb')
    for i in y:
        f.write(str(i)+'\n')
    f.close() 

def check_csr(feat):
    if scipy.sparse.isspmatrix_csr(feat):
        pass
    else:
        feat = scipy.sparse.csr_matrix(feat)
    return feat

def feature_transformer(emo, train_pos, train_neg, test_pos, test_neg):
    stopWords = stopwords.words('english')
#    X_train, X_test, y_train, y_test = format_data(emo, train_pos, train_neg, test_pos, test_neg)
    X_train, y_train = format_data(emo, train_pos, train_neg)
    X_test, y_test = format_data(emo, test_pos, test_neg)
    lexicon_feat = LexiconVectorizer()
    embed_feat = EmbeddingVectorizer()
    tfidf_feat = TfidfVectorizer(ngram_range=(1,3), analyzer='word', binary=False, stop_words=stopWords, min_df=0.01, use_idf=True)
    all_features = FeatureUnion([('lexicon_feature', lexicon_feat), ('embeddings', embed_feat), ('tfidf', tfidf_feat)])
#    Select = SelectFromModel(svm.LinearSVC(C=10, penalty="l1", dual=False))
#    scaler = StandardScaler(with_mean=False)
    pipeline = Pipeline([('all_feature', all_features)])
    feat_train = pipeline.fit_transform(X_train, y_train)
    feat_test = pipeline.transform(X_test)
    feat_size = feat_train.shape[1]
    return feat_train, y_train, feat_test, y_test, feat_size

def feature_transformer2(data,emo):
    stopWords = stopwords.words('english')
    data = [preprocess(data['text']).encode('utf-8')]
    lexicon_feat = LexiconVectorizer() # Globalise these two transformers
    embed_feat = EmbeddingVectorizer() # Globalise these two transformers
    tfidf_feat = TfidfVectorizer(ngram_range=(1,3), analyzer='word', binary=False, stop_words=stopWords, min_df=0.01, use_idf=True)
    all_features = FeatureUnion([('lexicon_feature', lexicon_feat), ('embeddings', embed_feat)])
    pipeline = Pipeline([('all_feature', all_features)])
    feat = pipeline.fit_transform(data)
    return feat

def main(train_pos, train_neg, test_pos, test_neg, d):
    for emo in train_pos.keys():
        print 80*"*"
        print "Feature extraction for emotion:", emo
        feat_train, y_train, feat_test, y_test, feat_size = feature_transformer(emo, train_pos, train_neg, test_pos, test_neg)
        feat_train = check_csr(feat_train)
        feat_test = check_csr(feat_test)
        print "Writing feature files"
        writevec('../output/'+d+'/'+emo+'/testing',feat_test,y_test)
        writevec('../output/'+d+'/'+emo+'/training',feat_train,y_train)
        writey('../output/'+d+'/'+emo+'/y_test', y_test)
        writey('../output/'+d+'/'+emo+'/y_train', y_train)
        print "Feature size =", feat_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--d", dest="d", help="all or sub or purverCV", default='all')
    args = parser.parse_args()
    train_pos, train_neg = load_tweets("purver_tweet_files_dict", "non_purver_tweet_files_dict")
    test_pos, test_neg = load_tweets("emotion_tweet_subset_dict", "non_emotion_tweet_subset_dict")
    main(train_pos, train_neg, test_pos, test_neg, args.d)
    
    
    