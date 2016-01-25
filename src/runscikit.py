import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from Utilities import load_tweets
from FeatureTransformer import feature_transformer


def predict(x_train, y_train, x_test, y_test, emo, ci):
    clf = svm.SVC(kernel='rbf', C=ci, class_weight='balanced')
    clf.fit(x_train, y_train)
#    joblib.dump(clf, '../models/'+emo+'.skl')
    y_predicted = clf.predict(x_test)
    print 'Precision score: ', metrics.precision_score(y_test, y_predicted, average='binary')
    print 'Recall score: ', metrics.recall_score(y_test, y_predicted, average='binary')
    print 'F1 score: ', metrics.f1_score(y_test, y_predicted, average='binary')


def main(train_pos, train_neg, test_pos, test_neg):
    for emo in train_pos.keys():
        print 80*"*"
        print "Emotion is", emo
        print "---Feature extraction"
        feat_train, y_train, feat_test, y_test, feat_size = feature_transformer(emo, train_pos, train_neg, test_pos, test_neg)
        print "Feature size =", feat_size
#        scaler = StandardScaler(with_mean=False)
#        scaler = MaxAbsScaler()
#        print "---Feature scaling"
#        feat_train_scaled = scaler.fit_transform(feat_train)
#        feat_test_scaled = scaler.transform(feat_test)
        print "---Model fitting and prediction"
        predict(feat_train, y_train, feat_test, y_test, emo, ci)
        print 80*"*"


if __name__ == "__main__":
    train_pos, train_neg = load_tweets("purver_tweet_files_dict", "non_purver_tweet_files_dict")
    test_pos, test_neg = load_tweets("emotion_tweet_subset_dict", "non_emotion_tweet_subset_dict")
    main(train_pos, train_neg, test_pos, test_neg)
