#!python3

'''

graph twitter sentiment, take two
put all three sections into one script?

get partied favour, by combing partisan words with sentiment
    ie sort tweets with trump or republican  vs pelosi or democrat
    then get sentiment for each
    assume democrat - = republican + and vice versa?

three areas:

create a twitter live feed which focuses on keyword
    this is mostly fine
    try to get data on what party is being adressed?
    learn more about this in general

create a sentiment analysis model
    this needs updating
    add more possible models
    fix rule-based model
    make easier to import into other scripts
    general clean up and optimization

'''


import nltk
import random
import pickle
from sklearn.svm import NuSVC
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier as sk
from sklearn.linear_model import LogisticRegression, SGDClassifier

import json
from tweepy import Stream
from Twitter_API_Keys import *  # locational of twitter keys, kept out for security
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

pos_txt_dir = 'positive.txt'
neg_txt_dir = 'negative.txt'

save_model_dir = 'SVA_model_2.pickle'
sentiment_save_dir = 'shutdown_sentiment_data.txt'

keywords = ['government shutdown']

# add keywords to see what party the sentiment is towards
dem_keywords = ['democrat', 'democrats', 'pelosi', 'dems', 'schumer', 'obama', 'mueller']
rep_keywords = ['republican', 'republicans', 'trump', 'mcconnell']


def get_features(text):
    tweet = word_tokenize(text)
    features = {}
    for word in word_list:  # iterating through whole allowed word_list seems inefficient...
        features[word] = (word in tweet)

    return features


def prepare_train_data():
    documents = []

    stop_words = set(stopwords.words('english'))
    # set up allowed word type? such as adjective, noun, verb?
    # nltk can mark these, and then allow them based on tags 'J', 'R', 'V'
    # these stand for adjective, adverb, noun, respectively
    # currently seems unnecessary, but can test accuracy changes later

    pos_text = open(pos_txt_dir, 'r').read()
    neg_text = open(neg_txt_dir, 'r').read()

    for doc in pos_text.split('\n'):
        documents.append((doc, 'pos'))

    for doc in neg_text.split('\n'):
        documents.append((doc, 'neg'))

    all_words = []

    pos_words = word_tokenize(pos_text)
    neg_words = word_tokenize(neg_text)

    for word in pos_words:
        if word not in stop_words:
            all_words.append(word.lower())

    for word in neg_words:
        if word not in stop_words:
            all_words.append(word.lower())

    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:5000]   # use how many of the most common words?

    def find_features(document):                # use this in other places? pull out of this function?
        words = set(word_tokenize(document))
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    feature_sets = [(find_features(doc), category) for (doc, category) in documents]

    random.shuffle(feature_sets)

    training_set = feature_sets[:10000]
    testing_set = feature_sets[10000:]

    return training_set, testing_set, word_features


class RuleCLF:
    # intake words
    # check if words in features in list of predetermined words

    # stick these in init?
    good_words = {'good', 'nice', 'happy', 'love',
                  'admirable', 'admire', 'admirably',
                  'advanced', 'altruistic', 'ambitious',
                  'applaud', 'appreciate', 'commend',
                  'congratulate', 'excellant', 'feat',
                  'foolproof', 'glory', 'god-given',
                  'god', 'god-send', 'handily', 'cunning',
                  'smart', 'clever', 'strong',
                  'noble', 'protect', 'fix', 'safe', 'jobs', 'improvement'}
    
    bad_words = {'bad', 'dumb', 'idiot', 'selfish',
                 'abrasive', 'accost', 'rude', 'alarm',
                 'angry', 'appalled', 'arrogance',
                 'ashamed', 'attack', 'awful', 'biased',
                 'brainless', 'brainwash', 'bribery',
                 'broken', 'brute', 'chaos', 'chaotic',
                 'childish', 'clueless', 'con', 'costly',
                 'counterproductive', 'counter-productive',
                 'cranky', 'damage', 'harm', 'resign'}

    def classify(self, features):
        pos_count = 0
        neg_count = 0
        # features will be a dictionary of word: bool(present)

        for word, appears in features.items():
            if appears:
                if word in self.good_words:
                    pos_count += 1
                elif word in self.bad_words:
                    neg_count += 1

        if pos_count > neg_count:
            return 'pos'
        elif neg_count > pos_count:
            return 'neg'
        else:
            return 'neutral'


class VoteClassifier:
    # intake a set of models
    # create a classify function that works via voting
    def __init__(self, *classifiers):
        self.classifiers = classifiers

    def classify(self, features):
        votes = []
        for clf in self.classifiers:
            votes.append(clf.classify(features))

        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / len(votes)
        # confidence inefficient?

        return vote_result, confidence


def train_model():
    # prep data and get train, test and word sets
    # train each model individually and print accuracy
    # put them all together in a SentimentAnalysisModel
    # return model and word_list
    train_data, test_data, word_set = prepare_train_data()
    
    bernoulli = sk(BernoulliNB())
    multinomial = sk(MultinomialNB())
    log_regr = sk(LogisticRegression())
    svc = sk(NuSVC())

    rule_clf = RuleCLF()

    bernoulli.train(train_data)
    multinomial.train(train_data)
    log_regr.train(train_data)
    svc.train(train_data)

    print('bernoulli', nltk.classify.accuracy(bernoulli, test_data))
    print('multinomial', nltk.classify.accuracy(multinomial, test_data))
    print('log_regr', nltk.classify.accuracy(log_regr, test_data))
    print('svc', nltk.classify.accuracy(svc, test_data))

    classifier = VoteClassifier(rule_clf,
                                bernoulli,
                                multinomial,
                                log_regr,
                                svc)
    
    return classifier, word_set


def sentiment(text):
    features = get_features(text)
    return model.classify(features)


def party_target(text):
    dem_votes = 0
    rep_votes = 0
    for word in word_tokenize(text):
        if word in dem_keywords:
            dem_votes += 1
        elif word in rep_keywords:
            rep_votes += 1
    if dem_votes > rep_votes:
        return 'd'
    elif rep_votes > dem_votes:
        return 'r'
    else:
        return ''


class Listener(StreamListener):
    # same but maybe add in checking for party keywords?
    def on_data(self, data):
        all_data = json.loads(data)
        confidence = 0
        try:
            tweet = all_data['text']
            tweet = tweet.lower()
            party = party_target(tweet)
            sentiment_value, confidence = sentiment(tweet)
            if not party == '':
                print(tweet, sentiment_value, confidence)
        except:
            pass

        if confidence >= 0.8 and not party == '':
            output = open(sentiment_save_dir, 'a')
            output.write(party + sentiment_value)
            output.write('\n')
            output.close()
            
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    '''

    need to multithread animation and twitter stream
    three solutions:
        publish subscribe pattern
            not quite right in this situation
        wrap both objects in functions and multithread the functions
            must be run through command prompt which causes other issues
        throw animation in separate function

    restructure so if not main, it just loads a model
    make it usable as an import module

    '''

    model, word_list = train_model()
    # save model and word_list?

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    twitterStream = Stream(auth, Listener())
    twitterStream.filter(track=keywords)
