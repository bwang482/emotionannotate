from sklearn.base import BaseEstimator
import numpy as np
import re

pos_emoticon = []
neg_emoticon = []
file = open('../lexicons/EmoticonSentimentLexicon.txt', 'r')
for line in file:
    if float(line.strip().split('\t')[1]) > 0:
        pos_emoticon.append(line.strip().split('\t')[0])
    elif float(line.strip().split('\t')[1]) < 0:
        neg_emoticon.append(line.strip().split('\t')[0])


class LexiconVectorizer(BaseEstimator):
    def get_feature_names(self):
        return self

    def fit(self, documents, y=None):
        return self

    def __init__(self):
        self.afinn_pos = {}
        self.afinn_neg = {}
        file01 = open('../lexicons/AFINN-111.txt', 'r')
        for each_line in file01:
            if float(each_line.strip().split('\t')[1]) > 0:
                self.afinn_pos[each_line.strip().split('\t')[0]] = float(each_line.strip().split('\t')[1])
            elif float(each_line.strip().split('\t')[1]) < 0:
                self.afinn_neg[each_line.strip().split('\t')[0]] = float(each_line.strip().split('\t')[1])
        file01.close()

        self.gi_pos = {}
        self.gi_neg = {}
        file02 = open('../lexicons/GeneralInquirer.txt', 'r')
        for each_line in file02:
            if float(each_line.strip().split('\t')[1]) > 0:
                self.gi_pos[each_line.strip().split('\t')[0]] = float(each_line.strip().split('\t')[1])
            elif float(each_line.strip().split('\t')[1]) < 0:
                self.gi_neg[each_line.strip().split('\t')[0]] = float(each_line.strip().split('\t')[1])
        file02.close()

        self.anticipation = {}
        self.anger = {}
        self.disgust = {}
        self.fear = {}
        self.joy = {}
        self.sadness = {}
        self.surprise = {}
        self.trust = {}
        for i in [line.strip().split('\t') for line in open("../lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2.txt")]:
            if i[0] == 'anticipation':
                self.anticipation[i[1]] = float(i[2])
            if i[0] == 'anger':
                self.anger[i[1]] = float(i[2])
            if i[0] == 'disgust':
                self.disgust[i[1]] = float(i[2])
            if i[0] == 'fear':
                self.fear[i[1]] = float(i[2])
            if i[0] == 'joy':
                self.joy[i[1]] = float(i[2])
            if i[0] == 'sadness':
                self.sadness[i[1]] = float(i[2])
            if i[0] == 'surprise':
                self.surprise[i[1]] = float(i[2])
            if i[0] == 'trust':
                self.trust[i[1]] = float(i[2])

        self.lexicon1_pos = {}
        self.lexicon1_neg = {}
        file1 = open('../lexicons/lexicon.txt', 'r')
        for each_line in file1:
            if float(each_line.strip().split()[1]) > 0:
                self.lexicon1_pos[each_line.strip().split()[0]] = float(each_line.strip().split()[1])
            elif float(each_line.strip().split()[1]) < 0:
                self.lexicon1_neg[each_line.strip().split()[0]] = float(each_line.strip().split()[1])
        file1.close()

        self.lexicon2_pos = []
        self.lexicon2_neg = []
        file2_pos = open('../lexicons/positive-words.txt', 'r')
        for each_line in file2_pos:
            self.lexicon2_pos.append(each_line.strip())
        file2_neg = open('../lexicons/negative-words.txt', 'r')
        for each_line in file2_neg:
            self.lexicon2_neg.append(each_line.strip())
        file2_pos.close()
        file2_neg.close()

        self.lexicon3_pos = {}
        self.lexicon3_neg = {}
        file3 = open('../lexicons/mpqa_05.tff', 'r')
        for each_line in file3:
            items = each_line.strip().split()
            match = re.search('word1=(\S+)', items[2])
            if match:
                if items[5] == 'priorpolarity=positive' and items[0] == 'type=weaksubj':
                    self.lexicon3_pos[match.group(1)] = float(1)
                if items[5] == 'priorpolarity=negative' and items[0] == 'type=weaksubj':
                    self.lexicon3_neg[match.group(1)] = float(1)
                if items[5] == 'priorpolarity=positive' and items[0] == 'type=strongsubj':
                    self.lexicon3_pos[match.group(1)] = float(2)
                if items[5] == 'priorpolarity=negative' and items[0] == 'type=strongsubj':
                    self.lexicon3_neg[match.group(1)] = float(2)
        file3.close()

        self.unihashtag_pos = {}
        self.unihashtag_neg = {}
        file4 = open('../lexicons/unigrams-pmilexicon_hashtag.txt', 'r')
        for each_line in file4:
            items = each_line.strip().split()
            if float(items[1]) > 0:
                self.unihashtag_pos[items[0]] = tuple(np.array(items)[[1, 2]])
            elif float(items[1]) < 0:
                self.unihashtag_neg[items[0]] = tuple(np.array(items)[[1, 3]])
        file4.close()

        self.bihashtag_pos = {}
        self.bihashtag_neg = {}
        file5 = open('../lexicons/bigrams-pmilexicon_hashtag.txt', 'r')
        for each_line in file5:
            items = each_line.strip().split('\t')
            if float(items[1]) > 0:
                self.bihashtag_pos[tuple(items[0].split())] = tuple(np.array(items)[[1, 2]])
            elif float(items[1]) < 0:
                self.bihashtag_neg[tuple(items[0].split())] = tuple(np.array(items)[[1, 3]])
        file5.close()

        self.unisenti140_pos = {}
        self.unisenti140_neg = {}
        file6 = open('../lexicons/unigrams-pmilexicon_sentiment140.txt', 'r')
        for each_line in file6:
            items = each_line.strip().split()
            if float(items[1]) > 0:
                self.unisenti140_pos[items[0]] = tuple(np.array(items)[[1, 2]])
            elif float(items[1]) < 0:
                self.unisenti140_neg[items[0]] = tuple(np.array(items)[[1, 3]])
        file6.close()

        self.bisenti140_pos = {}
        self.bisenti140_neg = {}
        file7 = open('../lexicons/bigrams-pmilexicon_sentiment140.txt', 'r')
        for each_line in file7:
            items = each_line.strip().split('\t')
            if float(items[1]) > 0:
                self.bisenti140_pos[tuple(items[0].split())] = tuple(np.array(items)[[1, 2]])
            elif float(items[1]) < 0:
                self.bisenti140_neg[tuple(items[0].split())] = tuple(np.array(items)[[1, 3]])
        file7.close()

        self.pairhashtag_pos = {}
        self.pairhashtag_neg = {}
        file8 = open('../lexicons/pairs-pmilexicon_hashtag.txt', 'r')
        for each_line in file8:
            items = each_line.strip().split('\t')
            if float(items[1]) > 0:
                self.pairhashtag_pos[frozenset(items[0].split('---'))] = tuple(np.array(items)[[1, 2]])
            elif float(items[1]) < 0:
                self.pairhashtag_neg[frozenset(items[0].split('---'))] = tuple(np.array(items)[[1, 3]])
        file8.close()

        self.pairsenti140_pos = {}
        self.pairsenti140_neg = {}
        file9 = open('../lexicons/pairs-pmilexicon_sentiment140.txt', 'r')
        for each_line in file9:
            items = each_line.strip().split('\t')
            if float(items[1]) > 0:
                self.pairsenti140_pos[frozenset(items[0].split('---'))] = tuple(np.array(items)[[1, 2]])
            elif float(items[1]) < 0:
                self.pairsenti140_neg[frozenset(items[0].split('---'))] = tuple(np.array(items)[[1, 3]])
        file9.close()

    def transform(self, documents):
        posemoticon = []
        negemoticon = []
        poslexicon1 = []
        neglexicon1 = []
        poslexicon2 = []
        neglexicon2 = []
        poslexicon3 = []
        neglexicon3 = []
        posunihashtag = []
        negunihashtag = []
        posunisenti140 = []
        negunisenti140 = []
        posbihashtag = []
        negbihashtag = []
        posbisenti140 = []
        negbisenti140 = []
        pospairhashtag = []
        negpairhashtag = []
        pospairsenti140 = []
        negpairsenti140 = []
        afinn_pos_lexicon = []
        afinn_neg_lexicon = []
        gi_pos_lexicon = []
        gi_neg_lexicon = []
        anticipation_lexicon = []
        anger_lexicon = []
        disgust_lexicon = []
        fear_lexicon = []
        joy_lexicon = []
        sadness_lexicon = []
        surprise_lexicon = []
        trust_lexicon = []
        for tweet in documents:
            d = tweet.lower()
            posemoticon_num = 0
            negemoticon_num = 0
            poslexicon1_num = 0
            neglexicon1_num = 0
            poslexicon2_num = 0
            neglexicon2_num = 0
            poslexicon3_num = 0
            neglexicon3_num = 0
            posunihashtag_num = 0
            negunihashtag_num = 0
            posunisenti140_num = 0
            negunisenti140_num = 0
            posbihashtag_num = 0
            negbihashtag_num = 0
            posbisenti140_num = 0
            negbisenti140_num = 0
            pospairhashtag_num = 0
            negpairhashtag_num = 0
            pospairsenti140_num = 0
            negpairsenti140_num = 0

            for each_emoticon in pos_emoticon:
                posemoticon_num += d.count(each_emoticon)
            for each_emoticon in neg_emoticon:
                negemoticon_num += d.count(each_emoticon)
            posemoticon.append(posemoticon_num)
            negemoticon.append(negemoticon_num)

            words = d.split()
            bigrams = [b for b in zip(d.split()[:-1], d.split()[1:])]

            afinn_pos_num = sum(map(lambda word: self.afinn_pos.get(word, 0), words))
            afinn_neg_num = sum(map(lambda word: self.afinn_neg.get(word, 0), words))
            gi_pos_num = sum(map(lambda word: self.gi_pos.get(word, 0), words))
            gi_neg_num = sum(map(lambda word: self.gi_neg.get(word, 0), words))
            anticipation_num = sum(map(lambda word: self.anticipation.get(word, 0), words))
            anger_num = sum(map(lambda word: self.anger.get(word, 0), words))
            disgust_num = sum(map(lambda word: self.disgust.get(word, 0), words))
            fear_num = sum(map(lambda word: self.fear.get(word, 0), words))
            joy_num = sum(map(lambda word: self.joy.get(word, 0), words))
            sadness_num = sum(map(lambda word: self.sadness.get(word, 0), words))
            surprise_num = sum(map(lambda word: self.surprise.get(word, 0), words))
            trust_num = sum(map(lambda word: self.trust.get(word, 0), words))

            for each_word in words:
                if each_word in self.lexicon1_pos:
                    poslexicon1_num += self.lexicon1_pos[each_word]
                elif each_word in self.lexicon1_neg:
                    neglexicon1_num += self.lexicon1_neg[each_word]

                if each_word in self.lexicon2_pos:
                    poslexicon2_num += 1
                elif each_word in self.lexicon2_neg:
                    neglexicon2_num += 1

                if each_word in self.lexicon3_pos:
                    poslexicon3_num += self.lexicon3_pos[each_word]

                if each_word in self.lexicon3_neg:
                    neglexicon3_num += self.lexicon3_neg[each_word]

                if each_word in self.unihashtag_pos:
                    posunihashtag_num += float(self.unihashtag_pos[each_word][0])
                elif each_word in self.unihashtag_neg:
                    negunihashtag_num += float(self.unihashtag_neg[each_word][0])

                if each_word in self.unisenti140_pos:
                    posunisenti140_num += float(self.unisenti140_pos[each_word][0])
                elif each_word in self.unisenti140_neg:
                    negunisenti140_num += float(self.unisenti140_neg[each_word][0])

            anticipation_lexicon.append(anticipation_num)
            anger_lexicon.append(anger_num)
            disgust_lexicon.append(disgust_num)
            fear_lexicon.append(fear_num)
            joy_lexicon.append(joy_num)
            sadness_lexicon.append(sadness_num)
            surprise_lexicon.append(surprise_num)
            trust_lexicon.append(trust_num)
            afinn_pos_lexicon.append(afinn_pos_num)
            afinn_neg_lexicon.append(afinn_neg_num)
            gi_pos_lexicon.append(gi_pos_num)
            gi_neg_lexicon.append(gi_neg_num)
            poslexicon1.append(poslexicon1_num)
            neglexicon1.append(neglexicon1_num)
            poslexicon2.append(poslexicon2_num)
            neglexicon2.append(neglexicon2_num)
            try:
                poslexicon3.append(float(poslexicon3_num)/len(words))
                neglexicon3.append(float(neglexicon3_num)/len(words))
            except:
                print d
                raise
            posunihashtag.append(posunihashtag_num/len(words))
            negunihashtag.append(negunihashtag_num/len(words))
            posunisenti140.append(posunisenti140_num/len(words))
            negunisenti140.append(negunisenti140_num/len(words))

            for each_bigram in bigrams:
                if each_bigram in self.bihashtag_pos:
                    posbihashtag_num += float(self.bihashtag_pos[each_bigram][0])
                elif each_bigram in self.bihashtag_neg:
                    negbihashtag_num += float(self.bihashtag_neg[each_bigram][0])

                if each_bigram in self.bisenti140_pos:
                    posbisenti140_num += float(self.bisenti140_pos[each_bigram][0])
                elif each_bigram in self.bisenti140_neg:
                    negbisenti140_num += float(self.bisenti140_neg[each_bigram][0])

            posbihashtag.append(posbihashtag_num/len(words))
            negbihashtag.append(negbihashtag_num/len(words))
            posbisenti140.append(posbisenti140_num/len(words))
            negbisenti140.append(negbisenti140_num/len(words))

            for each_item1 in words:
                for each_item2 in words:
                    if frozenset([each_item1, each_item2]) in self.pairhashtag_pos:
                        pospairhashtag_num += float(self.pairhashtag_pos[frozenset([each_item1, each_item2])][0])

                    elif frozenset([each_item1, each_item2]) in self.pairhashtag_neg:
                        negpairhashtag_num += float(self.pairhashtag_neg[frozenset([each_item1, each_item2])][0])

                    if frozenset([each_item1, each_item2]) in self.pairsenti140_pos:
                        pospairsenti140_num += float(self.pairsenti140_pos[frozenset([each_item1, each_item2])][0])

                    elif frozenset([each_item1, each_item2]) in self.pairsenti140_neg:
                        negpairsenti140_num += float(self.pairsenti140_neg[frozenset([each_item1, each_item2])][0])

            for each_item1 in bigrams:
                for each_item2 in bigrams:
                    if frozenset([' '.join(each_item1), ' '.join(each_item2)]) in self.pairhashtag_pos:
                        pospairhashtag_num += float(self.pairhashtag_pos[frozenset([' '.join(each_item1), ' '.join(each_item2)])][0])

                    elif frozenset([' '.join(each_item1), ' '.join(each_item2)]) in self.pairhashtag_neg:
                        negpairhashtag_num += float(self.pairhashtag_neg[frozenset([' '.join(each_item1), ' '.join(each_item2)])][0])

                    if frozenset([' '.join(each_item1), ' '.join(each_item2)]) in self.pairsenti140_pos:
                        pospairsenti140_num += float(self.pairsenti140_pos[frozenset([' '.join(each_item1), ' '.join(each_item2)])][0])

                    elif frozenset([' '.join(each_item1), ' '.join(each_item2)]) in self.pairsenti140_neg:
                        negpairsenti140_num += float(self.pairsenti140_neg[frozenset([' '.join(each_item1), ' '.join(each_item2)])][0])

            for each_item1 in words:
                for each_item2 in bigrams:
                    if frozenset([each_item1, ' '.join(each_item2)]) in self.pairhashtag_pos:
                        pospairhashtag_num += float(self.pairhashtag_pos[frozenset([each_item1, ' '.join(each_item2)])][0])

                    elif frozenset([each_item1, ' '.join(each_item2)]) in self.pairhashtag_neg:
                        negpairhashtag_num += float(self.pairhashtag_neg[frozenset([each_item1, ' '.join(each_item2)])][0])

                    if frozenset([each_item1, ' '.join(each_item2)]) in self.pairsenti140_pos:
                        pospairsenti140_num += float(self.pairsenti140_pos[frozenset([each_item1, ' '.join(each_item2)])][0])

                    elif frozenset([each_item1, ' '.join(each_item2)]) in self.pairsenti140_neg:
                        negpairsenti140_num += float(self.pairsenti140_neg[frozenset([each_item1, ' '.join(each_item2)])][0])

            pospairhashtag.append(pospairhashtag_num)
            negpairhashtag.append(negpairhashtag_num)
            pospairsenti140.append(pospairsenti140_num)
            negpairsenti140.append(negpairsenti140_num)

        result = np.array([posemoticon, negemoticon, afinn_pos_lexicon, afinn_neg_lexicon, gi_pos_lexicon, gi_neg_lexicon, anticipation_lexicon, anger_lexicon, disgust_lexicon, fear_lexicon, joy_lexicon, sadness_lexicon, surprise_lexicon, trust_lexicon, poslexicon1, neglexicon1, poslexicon2, neglexicon2, poslexicon3, neglexicon3, posunihashtag, negunihashtag, posunisenti140, negunisenti140, posbihashtag, negbihashtag, posbisenti140, negbisenti140, pospairhashtag, negpairhashtag, pospairsenti140, negpairsenti140]).T

        return result
