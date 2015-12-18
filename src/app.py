from flask import Flask
from flask import render_template, abort, jsonify, request,redirect, json
from Classifier import classifier
app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learning', methods=['POST'])
def learning():
    data = json.loads(request.data)
    # data == {"userInput": "whatever text you entered"}
    response = classifier(data)
    return jsonify(response)


if __name__ == '__main__':
    app.run(port = 9000)