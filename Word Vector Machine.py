#!python
# larger word to vec example
# text is...?

'''

use base of simple

intake a text                    what text?
clean up the text
    convert to lowercase
    remove stopwords
    stem??
    remove unneccessary symbols or mistaken words
    break into sentences using nltk
    break into words using nltk

generate set of features and lables using plain text

count number of unique words used in examples
convert features and labels to one hot

set up tensorflow model
train  using examples

generate a new dictionary using tf embeddings
write a function to find which words are closest to eachother

'''

import nltk
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

stemmer = nltk.stem.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))
# add stuff to stopwords?
# add punctiation to stop words so it stops getting associated?
puncts = ['.', ',', ';', ':', '===', '(', ')']
stop_words.update(puncts)
embedding_size = 128
iterations = 50000

data_dir = "industrial_wiki_pages.txt"
save_dir = "industrial_wiki_word_vecs.pickle"


def organize_text(text):
    text = text.lower()
    new_text = []
    corpus = nltk.tokenize.sent_tokenize(text)      # stopwords not getting removed properly
    print(len(corpus))
    for sentence in corpus:
        tmp = nltk.tokenize.word_tokenize(sentence)
        new_sent = []
        for word in tmp:
            if word not in stop_words:
                new_sent.append(word)
        new_text.append(new_sent)
    return new_text


def training_examples(text, examples=8500):     # find a way to stop the inclusion of 'words' that are just punctuation
    # increase window size?
    # more elegant way to do this?
    labels = []
    features = []
    for i in range(examples):
        sentence = random.choice(text)
        if len(sentence) < 2:
            continue
        pos = random.randint(0, len(sentence) - 1)
        label = sentence[pos]
        if pos > 0:
            feature = sentence[pos-1]
            labels.append(label)
            features.append(feature)
        if pos + 1 < len(sentence):
            feature = sentence[pos+1]
            labels.append(label)
            features.append(feature)
    labels = np.asarray(labels)
    features = np.asarray(features)
    return labels, features


def get_uniques(labels, features):
    # intake set of features and labels used
    uniques = []
    for i in range(len(labels)):
        uniques.append(labels[i])
        uniques.append(features[i])
    uniques = list(set(uniques))
    return uniques


def one_hot(word_set):
    word2vec = {}
    size = len(word_set)
    for i in range(len(word_set)):
        word = word_set[i]
        vector = np.zeros(size, dtype=np.int8)  # changing this to all ones and with target of 2 might reduce loss nan issues?
        vector[i] = 1
        vector = vector
        word2vec[word] = vector
    return word2vec


def convert_train_data(labels, features):
    label_set = []
    feature_set = []
    for label in labels:
        label_set.append(word_to_vec[label])
    for feature in features:
        feature_set.append(word_to_vec[feature])
    return label_set, feature_set


def closest(target, n_count=5):
    # calculate all distances
    # sort distances
    # return n_count closest
    vector = word_to_vec[target]
    distances = []
    for word in used_words:
        new_vec = word_to_vec[word]
        distance = np.linalg.norm(vector - new_vec)
        distances.append([distance, word])
    words = [i for i in sorted(distances)[:n_count]]
    return words
    
    


data_file = open(data_dir, encoding='utf-8')
text_data = data_file.read()
data_file.close()

text_data = organize_text(text_data)    # this can later be its own script?

y_train, x_train = training_examples(text_data)

used_words = get_uniques(x_train, y_train)
word_count = len(used_words)
print(word_count)                                   # when word count is odd, loss is nan?  also when word count is too high in general
word_to_vec = one_hot(used_words)
y_train, x_train = convert_train_data(y_train, x_train)

x_temp = tf.placeholder(tf.float32, shape=(None, word_count))
y_temp = tf.placeholder(tf.float32, shape=(None, word_count))

w1 = tf.Variable(tf.random_normal([word_count, embedding_size]))
b1 = tf.Variable(tf.random_normal([1]))

w2 = tf.Variable(tf.random_normal([embedding_size, word_count]))
b2 = tf.Variable(tf.random_normal([1]))

hidden_layer = tf.add(tf.matmul(x_temp, w1), b1) 
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, w2), b2))


#loss = tf.reduce_mean(-tf.reduce_sum(y_temp * tf.log(prediction), axis=[1]))        # sometimes loss is nan, higher values for
loss = tf.reduce_mean(tf.reduce_sum((y_temp - prediction)**2, axis=[1]))    # smaller loss means you need a higher learning rate for same change?


train_op = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

assert not np.any(np.isnan(x_train))
assert not np.any(np.isnan(y_train))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

##for iteration in range(iterations):
##    sess.run(train_op, feed_dict={x_temp: x_train, y_temp: y_train})
##    if iteration % 1000 == 0:
##        print(100 * iteration/iterations, '%, loss is: ',
##              sess.run(loss, feed_dict={x_temp:  x_train, y_temp: y_train}))
##print(100 * iteration/iterations, '%, loss is: ',
##      sess.run(loss, feed_dict={x_temp:  x_train, y_temp: y_train}))
##vectors = sess.run(w1 + b1)     # does removing bias help?

change = 10
r = 0
prev_loss = 10
while change > 0.001:
    r += 1
    for i in range(1000):
        sess.run(train_op, feed_dict={x_temp: x_train, y_temp: y_train})
    current_loss = sess.run(loss, feed_dict={x_temp:  x_train, y_temp: y_train})
    change = prev_loss - current_loss
    print(r, "rounds, loss is:", current_loss, "change is:", change)
    prev_loss = current_loss
    
        
'''
break out some test data to measure overfitting?

train in thousands at a time
drop learning rate when learning slows to much
continue for a group of predetermined learning rates?

higher learn rate improves faster at first, but can plateau
overall a higher learning rate just did way better overall

final loss values after 10,000 rounds training with same network settings:
each seem to have an area they get to, even low rates will jump quickly towards normal value then slow down...
0.001 --> 18 - 18.5        much less accuracte but predictions also seemed to be some of the best...only a small dataset and personal bias on that though
0.005 --> 17
0.01  --> 15
0.05  --> 5-6              this seems fishy, overtrained? somewhat accurate predictions

'''

vectors = sess.run(w1 + b1)
for i in range(len(used_words)):
    word_to_vec[used_words[i]] = vectors[i]

print(closest('britain'))
for i in range(5):
    print(closest(random.choice(used_words)))

save_file = open(save_dir, 'wb')
pickle.dump(word_to_vec, save_file)
save_file.close()
print('saved')






