# emotionannotate
For the [SMILE project](http://www.culturesmile.org/)

#### Set up
*Python 2.7 with pip 7.1.2*

Use 'pip install -r requirements.txt' to install all relevant Python dependancies.
#### Run web app
cd src

python app.py
#### Run batch classification
*If you have a list of tweets that you want to run our emotion classifier on, you can:*

cd src

python Classification.py

*Then follow the instructions.*

*More info in src/readme.txt*

#### Input/output data format
Input and output are both in json format.
For example,

{"tweetid":"614912375288893440", "text":"@britishmuseum awesome museum"}

as an input entry,

{"tweetid":"614912375288893440", 
"text":"@britishmuseum awesome museum",
"emotions":{"anger":"no","disgust":"no","happy":"yes","sad":"no","surprise":"no"}
}

as its output.


#### Reference

* Bo Wang, Maria Liakata, Arkaitz Zubiaga, Rob Procter and Eric Jensen. [SMILE: Twitter Emotion Classification using Domain Adaptation](http://ceur-ws.org/Vol-1619/paper3.pdf). In 4th Workshop on Sentiment Analysis where AI meets Psychology (SAAIP), IJCAI 2016.