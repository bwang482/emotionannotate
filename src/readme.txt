'app.py' runs a web app on 'localhost:9000' that allows you to enter a tweet and receive its emotion classification results.

'runlibsvmCV.py' is for cross-validating on a training data set, using libSVM.

'runlibsvm.py' is for tuning and fitting libsvm models and their parameters, and performance evaluation using existing training data set and your testing data.

'runasvm.py' is for tuning and fitting adaptive svm models and their parameters, and thus evaluating the performance.

'runscikit.py' uses scikit version of SVM.

'Classification.py' uses the same classifier as 'app.py' but you can submit multiple tweets to be processed as one single job. 
    - <input directory> is the directory that stalls your input files. Note: your input files have to be in JSON format.
    - <output directory> is the directory that you want to write the classification results to. The output files are in JSON format too.

