import sys
import os
import codecs
import json
from argparse import ArgumentParser
from twokenize import tokenizeRawTweetText as tokenize
import Transformer

TWEET_START = "tweet-start"
TWEET_END = "tweet-end"
SPACE = ' '


def preprocess(tweet):
    abbv_dict = json.load(open("../other/abbreviations.json"))
    emo_lexica_dict = json.load(open("../other/emoticons.json"))
    for emoticon in emo_lexica_dict[u'emoticons']:
        abbv_dict[emoticon] = ' '
    for word in emo_lexica_dict[u'words']:
        abbv_dict[word] = ' '
    hash_transformer = Transformer.HashtagTransformer()
    sub_transformer = Transformer.SubstitutionTransformer(abbv_dict)
    preprocessor = Preprocessor([hash_transformer, sub_transformer])
    tweet = ' '.join(tokenize(tweet))
    tweet = preprocessor.transform(tweet)
    return tweet


class Preprocessor(object):
    """For a given set of transformers, read in a text file containing tweets and preprocess them"""
    def __init__(self, transformers):
        """transformers: a list of Transformers which will have tweets passed to them in order"""
        super(Preprocessor, self).__init__()
        self.transformers = transformers

    def transform(self, tweet):
        """For a given tweet, return the result of passing it through the Transformers"""
        for transformer in self.transformers:
            tweet = transformer.transform(tweet)
        return tweet

    def read_tweets(self, filename, emo):
        """Read tweets in raw format, returning a list of all tweets in the file"""
        emo_tweets = []
        non_emo_tweets = []
        with codecs.open(filename, encoding='utf8') as tweet_file:
#            tweet = []
            for line in tweet_file:
                data = json.loads(line)
                id = data['tweetid'].strip()
                text = data['text'].strip()
                emotions = data['emotions']
                tokens = tokenize(text)
                incount = 0
                for e in emotions:
                    if e == emo:
                        incount = 1
                if incount == 1:
                    emo_tweets.append(SPACE.join(tokens))
                elif incount == 0:
                    non_emo_tweets.append(SPACE.join(tokens))
        return emo_tweets, non_emo_tweets

    def preprocess(self, filename, emo):
        """Read and preprocess all tweets in a file in raw format, returning a list of tweets"""
        emo_tweets, non_emo_tweets = self.read_tweets(filename, emo)
        emotiontweets = []
        nonemotiontweets = []
        for tweet in emo_tweets:
            emotiontweets.append(self.transform(tweet))
        for tweet in non_emo_tweets:
            nonemotiontweets.append(self.transform(tweet))
        return emotiontweets, nonemotiontweets


if __name__ == '__main__':
    emos = ['happy', 'angry', 'disgust', 'sad', 'surprise']
    abbv_dict = json.load(open("../other/abbreviations.json"))
    emo_lexica_dict = json.load(open("../other/emoticons.json"))
    # Add emoticons from emotions.json
    for emoticon in emo_lexica_dict[u'emoticons']:
        abbv_dict[emoticon] = ' '
    # Add words from emotions.json
    for word in emo_lexica_dict[u'words']:
        abbv_dict[word] = ' '
    hash_transformer = Transformer.HashtagTransformer()
    sub_transformer = Transformer.SubstitutionTransformer(abbv_dict)
    preprocessor = Preprocessor([hash_transformer, sub_transformer])
    parser = ArgumentParser()
    parser.add_argument("--inputdir", dest="source_dir", help="input directory", default="../data/input")
    parser.add_argument("--outputdir", dest="output_dir", help="output directory", default="../data/preprocessed")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.source_dir) if f.endswith('.json')]
    for emo in emos:
        for filename in files:
            input_filename = os.path.join(args.source_dir, filename)
            print 'Preprocessing tweet for emotion category:', emo
            emotion_tweets, non_emotion_tweets = preprocessor.preprocess(input_filename, emo)
            with codecs.open(os.path.join(args.output_dir, 'emo_'+emo+'.txt'), "w+", encoding='utf-8') as output_file:
                output_file.write('\n'.join(emotion_tweets))
            with codecs.open(os.path.join(args.output_dir, 'emo_'+'non'+emo+'.txt'), "w+", encoding='utf-8') as output_file:
                output_file.write('\n'.join(non_emotion_tweets))
