# emotionannotate
For the SMILE project

#### Set up
Use 'pip install -r requirement.txt' to install all relevant Python dependancies.
#### Run web app
cd src

python app.py
#### Run multi-tweets emotion classification
If you have a list of tweets that you want to run our emotion classifier on, you can:

cd src

python Classification.py --inputdir <input directory> --outputdir <outpur directory>

More info in src/readme.txt

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
