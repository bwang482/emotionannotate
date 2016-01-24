import json, os
import codecs
import sys
from argparse import ArgumentParser

from Classifier import parallelClassifier, initFeatureProcessors

print "\nPlease wait until the initialisation is finished..."
global lexicon_feat, embed_feat
lexicon_feat = None
embed_feat = None

initFeatureProcessors()
def read_tweets(inputdir):
    files = [f for f in os.listdir(inputdir) if f.endswith('.json')]
    tweets = []
    for filename in files:
        input_filename = os.path.join(inputdir, filename)
        with codecs.open(input_filename, encoding='utf8') as tweet_file:
            for line in tweet_file:
                data = json.loads(line)
                tweets.append(data)
    return tweets
    
def write_results(outputdir, results):
    fp = open(outputdir+'/results.json', 'wb')
    for result in results:
        json.dump(result, fp)
        fp.write('\n')
    fp.close()

def classify(inputdir,outputdir):
    import datetime
    global lexicon_feat, embed_feat
    print datetime.datetime.now()
    tweets = read_tweets(inputdir)
    results = parallelClassifier(tweets, lexicon_feat, embed_feat)
    print datetime.datetime.now()
    write_results(outputdir, results)
    
    
if __name__ == '__main__':
    global lexicon_feat, embed_feat
    while True:
        if not lexicon_feat:
            lexicon_feat, embed_feat = initFeatureProcessors()
        elif not embed_feat:
            lexicon_feat, embed_feat = initFeatureProcessors()
        answer='yes'
        if answer.lower().startswith("y"):
            inputs = raw_input("\n\nPlease type your data input directory and results output directory here (with white space in between)\ne.g. /home/user/inputFolder home/user/Desktop/outputFolder\n(Type CTRL-C to exit)\n")
            dirTokens = inputs.split(" ")
            inputDir = dirTokens[0]
            outputDir = dirTokens[1]
            classify(inputDir, outputDir)
            answer = raw_input('Do you want to continue? ')
            if answer.lower().startswith("n"):
               print("Thanks for using our emotion classifier.")
               exit()
    
                