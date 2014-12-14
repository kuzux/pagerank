import re
from nltk.tokenize import RegexpTokenizer
from numpy import *


#########################################
#### Filtering out stop words        ####
#########################################
stop_words = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 
                  'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 
                  'because', 'been', 'before', 'being', 'below', 'between', 
                  'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 
                  'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 
                  'down', 'during', 'each', 'few', 'for', 'from', 'further', 
                  'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 
                  'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 
                  'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", 
                  "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', 
                  "it's", 'its', 'itself', "let's", 'me', 'more', 'most', 
                  "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 
                  'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
                  'ourselves', 'out', 'over', 'own', 'rt', 'same', "shan't", 
                  'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so',
                  'some', 'such', 'than', 'that', "that's", 'the', 'their',
                  'theirs', 'them', 'themselves', 'then', 'there', "there's",
                  'these', 'they', "they'd", "they'll", "they're", "they've", 
                  'this', 'those', 'through', 'to', 'too', 'under', 'until', 
                  'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", 
                  "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 
                  'where', "where's", 'which', 'while', 'who', "who's", 'whom', 
                  'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', 
                  "you'd", "you'll", "you're", "you've", 'your', 'yours', 
                  'yourself', 'yourselves'])

def is_stop(word):
    if word[0] == '@':
        return True
    elif word in stop_words: 
        return True
    else: 
        return False




###########################################################
#### Preprocess the tweets to get sets of word indices ####
#### a table of words to word indices, and its reverse ####
###########################################################

tokenizer = RegexpTokenizer(r"@?(\w+'\w+)|(\w+)")
tweets = []
word_indices = {}
all_words = []
curr_word_index = 0

with open("tweetset.txt", "r") as tweets_file:
    for line in tweets_file:
        words = tokenizer.tokenize(line)
        curr_tweet = set()
        for word in words:
            word = word.lower()
            if is_stop(word): continue
            if word in word_indices:
                curr_tweet.add(word_indices[word])
            else:
                word_indices[word] = curr_word_index
                all_words.append(word)
                curr_tweet.add(curr_word_index)
                curr_word_index += 1
        tweets.append(curr_tweet)




#######################################################
#### Constructing the adjacency matrix from tweets ####
#######################################################

num_words = curr_word_index
adj_matrix = zeros((num_words, num_words))
for tweet in tweets:
    for word1 in tweet:
        for word2 in tweet:
            if word1 == word2: continue
            adj_matrix[word1, word2] += 1
            adj_matrix[word2, word1] += 1





################################################################
#### Converting the adjacency matrix into transition matrix ####
#### by normalizing edge weights & adding teleportation     ####
################################################################

row_weights = sum(adj_matrix, axis=1)
adj_matrix /= row_weights[:, None]
trans_matrix = 0.86*adj_matrix + (0.14/num_words)*ones((num_words, num_words))

curr_vector = (ones(num_words)/num_words)
prev_vector = zeros(num_words)






#####################################################################
#### Calculate the page rank with power method                   ####
#### Iterating until the change is smaller than an epsilon value ####
#####################################################################

epsilon = 0.00001
i = 1
while linalg.norm(curr_vector-prev_vector) > epsilon:
    # print "iteration %d" % (i, )
    i += 1
    prev_vector = curr_vector
    curr_vector = curr_vector.dot(trans_matrix)





#############################################################
#### Obtain the page rank for each word in the tweet set ####
#############################################################

res = []
for i in range(num_words):
    res.append((curr_vector[i], all_words[i]))

res = sorted(res, reverse=True)





############################################################
#### Write the results to a file                        ####
############################################################

with open("results.txt", "w") as outfile:
    for score, word in res:
        outfile.write("%s %f\n" % (word, score))
