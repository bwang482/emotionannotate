import json, os
import codecs
from argparse import ArgumentParser

from Classifier import parallelClassifier

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
    for result in results:
        with open(outputdir+'/results.json', 'a') as fp:
            json.dump(result, fp)
            fp.write('\n')

def classify(inputdir,outputdir):
    tweets = read_tweets(inputdir)
    results = []
    for tweet in tweets:
        result = parallelClassifier(tweet)
        results.append(result)
    write_results(outputdir, results)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--inputdir", dest="source_dir", help="input directory")
    parser.add_argument("--outputdir", dest="output_dir", help="output directory")
    args = parser.parse_args()
    
    classify(args.source_dir,args.output_dir)
    
                