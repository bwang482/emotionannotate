from flask import Flask
from flask import render_template, abort, jsonify, request,redirect, json
from Classifier import parallelClassifier
app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learning', methods=['POST'])
def learning():
    data = json.loads(request.data)
    # try 'lucky @USERID ! good luck @USERID & see you soon :) @USERID @USERID'
    result = parallelClassifier([data])
    emotions = result[0]['emotions']
    return jsonify(emotions)


if __name__ == '__main__':
    app.run(port = 9000)