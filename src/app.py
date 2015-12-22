from flask import Flask
from flask import render_template, abort, jsonify, request,redirect, json
from Classifier import parallelClassifier, initFeatureProcessors
app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learning', methods=['POST'])
def learning():
    lexicon_feat, embed_feat = initFeatureProcessors()
    data = json.loads(request.data)
    # try 'lucky @USERID ! good luck @USERID & see you soon :) @USERID @USERID'
    result = parallelClassifier([data], lexicon_feat, embed_feat)
    emotions = result[0]['emotions']
    return jsonify(emotions)


if __name__ == '__main__':
    app.run(port = 9000)