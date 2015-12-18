
"""
    This module serves the purpose as a single location to store config information such as
    directories containing tweets, locations for saving and loading model files, number of folds in
    the cross validation tests etc. Once fully refactored this class will contain all user changable parameters
    within the program.
"""

# CHANGE PARAMETERS HERE #

emotion_tweet_dir = "../data/preprocessed/"
emotion_subset_dir = "../data/subset/"
purver_tweet_dir = "../data/purver_old/"
num_cv_folds = 10
emotion_model_name = "../saved_models/"

# END OF PARAMETER VARIABLES HERE #

param_dictionary = {
    "emotion_tweet_files_dict": {'anger': [emotion_tweet_dir + 'emo_angry.txt'],
               'disgust': [emotion_tweet_dir + 'emo_disgust.txt'],
#               'fear': [emotion_tweet_dir + 'emo_fear_raw.txt'],
               'happy': [emotion_tweet_dir + 'emo_happy.txt'],
               'sad': [emotion_tweet_dir + 'emo_sad.txt'],
               'surprise': [emotion_tweet_dir + 'emo_surprise.txt']},
    "non_emotion_tweet_files_dict": {'anger': [emotion_tweet_dir + 'emo_nonangry.txt'],
               'disgust': [emotion_tweet_dir + 'emo_nondisgust.txt'],
#               'fear': [emotion_tweet_dir + 'emo_nonfear_raw.txt'],
               'happy': [emotion_tweet_dir + 'emo_nonhappy.txt'],
               'sad': [emotion_tweet_dir + 'emo_nonsad.txt'],
               'surprise': [emotion_tweet_dir + 'emo_nonsurprise.txt']},
            
    "emotion_tweet_subset_dict": {'anger': [emotion_subset_dir + 'emo_angry.txt'],
               'disgust': [emotion_subset_dir + 'emo_disgust.txt'],
#               'fear': [emotion_subset_dir + 'emo_fear_raw.txt'],
               'happy': [emotion_subset_dir + 'emo_happy.txt'],
               'sad': [emotion_subset_dir + 'emo_sad.txt'],
               'surprise': [emotion_subset_dir + 'emo_surprise.txt']},
    "non_emotion_tweet_subset_dict": {'anger': [emotion_subset_dir + 'emo_nonangry.txt'],
               'disgust': [emotion_subset_dir + 'emo_nondisgust.txt'],
#               'fear': [emotion_subset_dir + 'emo_nonfear_raw.txt'],
               'happy': [emotion_subset_dir + 'emo_nonhappy.txt'],
               'sad': [emotion_subset_dir + 'emo_nonsad.txt'],
               'surprise': [emotion_subset_dir + 'emo_nonsurprise.txt']},
            
    "purver_tweet_files_dict": {'anger': [purver_tweet_dir + 'emo_anger_raw.txt', purver_tweet_dir + 'hash_anger_raw.txt'],
                   'disgust': [purver_tweet_dir + 'emo_disgust_raw.txt', purver_tweet_dir + 'hash_disgust_raw.txt'],
#                   'fear': [emotion_tweet_dir + 'emo_fear_raw.txt', emotion_tweet_dir + 'hash_fear_raw.txt'],
                   'happy': [purver_tweet_dir + 'emo_happy_raw.txt', purver_tweet_dir + 'hash_happy_raw.txt'],
                   'sad': [purver_tweet_dir + 'emo_sad_raw.txt', purver_tweet_dir + 'hash_sad_raw.txt'],
                   'surprise': [purver_tweet_dir + 'emo_surprise_raw.txt', purver_tweet_dir + 'hash_disgust_raw.txt']},
    "non_purver_tweet_files_dict": {'anger': [purver_tweet_dir + 'emo_nonanger_raw.txt', purver_tweet_dir + 'hash_nonanger_raw.txt'],
                   'disgust': [purver_tweet_dir + 'emo_nondisgust_raw.txt', purver_tweet_dir + 'hash_nondisgust_raw.txt'],
#                   'fear': [emotion_tweet_dir + 'emo_nonfear_raw.txt', emotion_tweet_dir + 'hash_nonfear_raw.txt'],
                   'happy': [purver_tweet_dir + 'emo_nonhappy_raw.txt', purver_tweet_dir + 'hash_nonhappy_raw.txt'],
                   'sad': [purver_tweet_dir + 'emo_nonsad_raw.txt', purver_tweet_dir + 'hash_nonsad_raw.txt'],
                   'surprise': [purver_tweet_dir + 'emo_nonsurprise_raw.txt', purver_tweet_dir + 'hash_nondisgust_raw.txt']},
            
    "num_cv_folds": num_cv_folds,
    "emotion_model_name": emotion_model_name,
}


def get(key):
    return param_dictionary[key]


# Pointless comment
