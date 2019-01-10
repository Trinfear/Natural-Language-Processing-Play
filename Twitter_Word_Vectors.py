#!python3
#  word vectors


import nltk
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.tokenize import word_tokenize

embedding_size = 1024
batch_size = 512

pos_data_dir = 'positive.txt'
neg_data_dir = 'negative.txt'
vector_save_dir = 'twitter_word_vectors.pickle'


def get_data():
    # open both files
    # split them into lines
    # add into a single set
    # tokenize each sentence?
    # remove stopwords?
    # return 2d array of sentences of words?
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update("'", '"', '.', ',', '?', '!')
    
    data_file = open(pos_data_dir)
    pos_data = data_file.read()
    data_file.close()

    data_file = open(pos_data_dir)
    neg_data = data_file.read()
    data_file.close()

    pos_data = pos_data.split('\n')
    neg_data = neg_data.split('\n')

    all_data = pos_data + neg_data

    processed_data = []
    for sentence in all_data:
        new_sent = []
        words = word_tokenize(sentence)
        for word in words:
            if word not in stop_words:
                new_sent.append(word)
        if len(new_sent) > 2:
            processed_data.append(new_sent)

    return processed_data


def get_uniques(corpus):
    # iterate through all sentences and find all unique words
    uniques = set()
    for sentence in corpus:
        uniques.update(set(sentence))
    return list(uniques)


def one_hot(words):
    # create one hot embedding for each unique word
    # return dictionary for word to embeddings
    # return dictionary
    word2vec = {}
    size = len(words)
    for i in range(size):
        word = words[i]
        vector = np.zeros(size, dtype=np.int8)
        vector[i] = 1
        word2vec[word] = vector

    return word2vec


def generate_examples(corpus, train_pct=0.95, train_size=35000,
                      test_size=5000):
    # roughly 13.3k words across 10.5k sentences
    # splits data into train and test
    # generates features and labels using corpus for each segment
        # use CBOW, ie feed context words to guess target word
    # return train and test features/labels
    random.shuffle(corpus)
    split_point = int(len(corpus) * train_pct)
    train_set = corpus[:split_point]
    test_set = corpus[split_point:]

    for i in range(train_size):
        # get target
        sentence = random.choice(train_set)
        position = random.randint(0, len(sentence)-1)
        label = sentence[position]
        
        # get features
        # create separate things for each word
        # or have all words together as a single feature?
        if position > 0:
            features = sentence[position - 1]
            train_set.append((features, label))
        if position < len(sentence) - 1:
            features = sentence[position + 1]
            train_set.append((features, label))

    for i in range(test_size):
        # get target
        sentence = random.choice(test_set)
        position = random.randint(0, len(sentence)-1)
        label = sentence[position]
        
        # get features
        # create separate things for each word
        # or have all words together as a single feature?
        if position > 0:
            features = sentence[position - 1]
            test_set.append((features, label))
        if position < len(sentence) - 1:
            features = sentence[position + 1]
            test_set.append((features, label))

    train_set = np.asarray(train_set)
    test_set = np.asarray(test_set)
    
    return train_set, test_set


def convert_data(data, dictionary):
    # convert words into vector embeddings using dictionary
    embedded_data = []
    for datum in data:
        feature = dictionary[datum[0]]
        label = dictionary[datum[1]]
        embedded_data.append((feature, label))
    return embedded_data


def generate_embeddings(embed_vector, vector_size, train_set, test_set,
                        uniques):
    # generate tf model
    # return model
    x_train = [j[0] for j in train_set]
    y_train = [j[1] for j in train_set]

    x_temp = tf.placeholder(tf.float32, shape=(None, vector_size))
    y_temp = tf.placeholder(tf.float32, shape=(None, vector_size))

    w1 = tf.Variable(tf.random_normal([vector_size, embed_vector]))
    b1 = tf.Variable(tf.random_normal([1]))  # why isn't this vector_size?

    w2 = tf.Variable(tf.random_normal([embed_vector, vector_size]))
    b2 = tf.Variable(tf.random_normal([1]))

    hidden_layer = tf.add(tf.matmul(x_temp, w1), b1)
    prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, w2), b2))

    # different error functions?  Standard seems to be the first one, but keeps returning nan
    # error = tf.reduce_mean(tf.reduce_sum(tf.square(y_temp - prediction), axis=[1]))
    # error = tf.reduce_mean(tf.reduce_sum((y_temp - prediction)**2, axis=[1]))
    error = tf.reduce_mean(-tf.reduce_sum(y_temp * tf.log(prediction), axis=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.02).minimize(error)
    # add in changing learning_rate when change drops below x, drop learning rate and keep going?

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    change = 100
    cycle = 0
    prev_loss = 100
    batches = int(len(train_set)/batch_size)
    batch_start = 0
    while change > 0.0001:
        cycle += 1
        for i in range(1000):
            for j in range(batches):
                batch_end = batch_start + batch_size
                if batch_end > len(x_train):
                    continue
                train_dict = {x_temp: x_train[batch_start:batch_end],
                              y_temp: y_train[batch_start:batch_end]}
                batch_start = batch_end
                sess.run(train_op, feed_dict=train_dict)
                
        current_loss = sess.run(error, feed_dict={x_temp: x_train[:batch_size],
                                                  y_temp: y_train[:batch_size]})
        change = prev_loss - current_loss
        
        print(cycle, ' rounds trained, loss is: ', current_loss, 'Change: ',
              change)
        
        prev_loss = current_loss
        batch_start = 0

    vectors = sess.run(w1 + b1)
    new_word_embeddings = {}
    for i in range(vector_size):
        new_word_embeddings[uniques[i]] = vectors[i]

    # use test set to get accuracy?
    # get loss for each example
    # take mean of loss?
    test_errors = []
    for datum in test_set:
        test_dict = {x_temp: [datum[0]], y_temp: [datum[1]]}
        test_error = sess.run(error, feed_dict=test_dict)
        test_errors.append(test_error)

    test_mean_error = sess.run(tf.reduce_mean(test_errors))
    print(test_mean_error)

    return new_word_embeddings


def closest_words(target, n_count=5):
    # calculate distances between other words
    # return closest word
    vector = new_embeddings[target]
    distances = []
    for word in unique_words:
        if word == target:
            continue
        new_vec = new_embeddings[word]
        distance = np.linalg.norm(vector - new_vec)
        distances.append((distance, word))
    close_words = sorted(distances)[:n_count]
    return close_words


if __name__ == '__main__':
    # intake train data
    # find all unique words
    # create one hot embedding for words
    # convert text to one hot embedding
    # generate model
    # train model and fetch embeddings
    # test model
    # save embeddings
    data_set = get_data()
    unique_words = get_uniques(data_set)
    one_hot_embeddings = one_hot(unique_words)
    print(len(data_set), len(unique_words))

    train_data, test_data = generate_examples(data_set)
    print(len(train_data), len(test_data))
    # data in the form of an np array of tuples (feature, label)

    train_data = convert_data(train_data, one_hot_embeddings)
    test_data = convert_data(test_data, one_hot_embeddings)

    print(train_data[1][1])

    new_embeddings = generate_embeddings(embedding_size, len(unique_words),
                                         train_data, test_data, unique_words)

    print(closest_words('friend'))
    print(closest_words('funny'))
    print(closest_words('think'))
    print(closest_words('fast'))
    print(closest_words('power'))

    save_file = open(vector_save_dir, 'wb')
    pickle.dump(new_embeddings, save_file)
    save_file.close()
    print('saved')
