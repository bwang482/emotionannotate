import json, os
import codecs, sys
from argparse import ArgumentParser

from Classifier import parallelClassifier, initFeatureProcessors

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
        print "Press enter to re-run the script, CTRL-C to exit"
        sys.stdin.readline()
        #reload(mainscript)
        parser = ArgumentParser()
        parser.add_argument("--inputdir", dest="source_dir", help="input directory")
        parser.add_argument("--outputdir", dest="output_dir", help="output directory")
        args = parser.parse_args()
        classify(args.source_dir,args.output_dir)
    
                