'app.py' runs a web app on 'localhost:9000' that allows you to enter a tweet and receive its emotion classification results.

'runlibsvmCV.py' is for cross-validating on a training data set, using libSVM.

'runlibsvm.py' is for parameters tuning, model fitting and performance evaluation using existing training data set and your testing data.

'runscikit.py' uses scikit version of SVM.

'Classification.py' uses the same classifier as 'app.py' but you can submit multiple tweets to be processed as one single job. 
    - Usage: 'python Classification.py --inputdir <input directory> --outputdir <outpur directory>
    - <input directory> is the directory that stalls your input files. Note: your input files have to be in JSON format.
    - <output directory> is the directory that you want to write the classification results to. The output files are in JSON format too.

