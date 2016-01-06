# emotionannotate
For the SMILE project

#### Set up
*Python 2.7 with pip 7.1.2*

Use 'pip install -r requirement.txt' to install all relevant Python dependancies.
#### Run web app
cd src

python app.py
#### Run batch classification
*If you have a list of tweets that you want to run our emotion classifier on, you can:*

cd src

python Classification.py --inputdir <input directory> --outputdir <outpur directory>

e.g. 'python Classification.py --inputdir ../input --outputdir ../output'

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

## Building the Development Environment

### Requirements

* [Ansible](https://github.com/ansible/ansible)
* [Vagrant](https://www.vagrantup.com)
* [VirtualBox](https://www.virtualbox.org)

### Starting up

From the project directory:
```shell
vagrant up
```
This will download the relevant Vagrant box, create the development environment and install the required packages. Once finished, you can use:
```shell
vagrant ssh
```
to connect to the box. To start the development server:
```shell
cd /vagrant/src/
python app.py
```
The system will be running on your machine. You can load the homepage in a web browser: http://127.0.0.1:9000
