# emotionannotate
For the SMILE project

#### Set up
Use 'pip install -r requirement.txt' to install all relevant Python dependancies.
#### Run web app
cd src
python app.py

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
