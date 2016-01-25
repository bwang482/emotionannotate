import numpy as np
import os
import Config
from sklearn.cross_validation import train_test_split


def class_dist(dir):
    emotions = ['anger', 'disgust', 'happy', 'surprise', 'sad']
    for emo in emotions:
        f = open(dir+'/emo_'+emo+'_raw.txt', 'r')
        lines = f.readlines()
        p1 = len(lines)
        f = open(dir+'/hash_'+emo+'_raw.txt', 'r')
        lines = f.readlines()
        p2 = len(lines)
        f = open(dir+'/hash_non'+emo+'_raw.txt', 'r')
        lines = f.readlines()
        n2 = len(lines)
        f = open(dir+'/emo_non'+emo+'_raw.txt', 'r')
        lines = f.readlines()
        n1 = len(lines)
        print float((p1+p2))/(p1+p2+n1+n2), float((n1+n2))/(p1+p2+n1+n2)


def load_tweets(emotion_tweet_dict, non_emotion_tweet_dict):
    # Load positive tweets from files to memory
    tweet_files_positive = Config.get(emotion_tweet_dict)
    positive_tweets = {}
    for emo in tweet_files_positive.keys():
        positive_tweets[emo] = []
        for filename in tweet_files_positive[emo]:
            file_obj = open(filename, "r")
            for tweet in file_obj:
                positive_tweets[emo].append(tweet)

    # Load negative tweets from files to memory
    tweet_files_negative = Config.get(non_emotion_tweet_dict)
    negative_tweets = {}
    for emo in tweet_files_negative.keys():
        negative_tweets[emo] = []
        for filename in tweet_files_negative[emo]:
            file_obj = open(filename, "r")
            for tweet in file_obj:
                negative_tweets[emo].append(tweet)
    return positive_tweets, negative_tweets


def format_data(emo, pos, neg):
    d = []
    for tweet in pos[emo]:
        d.append((tweet.strip(), 1))
    for tweet in neg[emo]:
        d.append((tweet.strip(), 0))
    X = []
    y = []
    for tweet, label in d:
        if tweet != '':
            X.append(tweet)
            y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def feval(truefile, predfile):
    truefile = os.path.abspath(truefile)
    predfile = os.path.abspath(predfile)
    f1 = open(truefile, 'r')
    f2 = open(predfile, 'r')
    l1 = f1.readlines()
    l2 = f2.readlines()
    y_test = []
    y_predicted = []
    if len(l1) == len(l2):
        for i in xrange(len(l1)):
            y_test.append(int(l1[i].strip()))
            y_predicted.append(int(l2[i].strip()))
    else:
        raise Exception('ERROR: true and pred file length do not match!')
    f1.close()
    f2.close()
    return y_test, y_predicted


def getlabels(X):
    y = []
    for i in X:
        i = i[0].split(' ')
        y.append(int(i[0]))
    return y


def writingfile(filepath, X):
    with open(filepath, 'w') as f:
        for item in X:
            f.write("%s\n" % item)


def readfeats(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    d = []
    for i in lines:
        d.append(i.strip())
    d = np.asarray(d)
    return d


def frange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r *= step


def subset_data(sub_dir='../data/subset/'):
    pos, neg = load_tweets("emotion_tweet_files_dict", "non_emotion_tweet_files_dict")
    for emo in pos.keys():
        d = []
        for tweet in pos[emo]:
            d.append((tweet.strip(), 1))
        for tweet in neg[emo]:
            d.append((tweet.strip(), 0))
        X = []
        y = []
        for tweet, label in d:
            X.append(tweet)
            y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        pos_test = []
        neg_test = []
        for i, j in enumerate(X_test):
            if y_test[i] == 1:
                pos_test.append(j)
            elif y_test[i] == 0:
                neg_test.append(j)
        pos_dir = sub_dir+'emo_'+emo+'.txt'
        neg_dir = sub_dir+'emo_non'+emo+'.txt'
        writingfile(pos_dir, pos_test)
        writingfile(neg_dir, neg_test)
