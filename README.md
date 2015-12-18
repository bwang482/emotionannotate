# emotionannotate
For the SMILE project

cd src

python app.py


Input and output are both in json format.

For example,

{"tweetid":"614912375288893440", "text":"@britishmuseum awesome museum"}

as a input,

{"tweetid":"614912375288893440", 
"text":"@britishmuseum awesome museum",
"emotions":{"anger":"no","disgust":"no","happy":"yes","sad":"no","surprise":"no"}
}

as its output.