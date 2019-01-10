# Natural-Language-Processing-Play
Various scripts playing around with natural language processing techniques and uses.
Currently contains a script to pull data from a related time period from wikipedia, all focused roughly around the industrial revolution.
This data is used to generate a set of word embeddings, which may be used in other algorithms, such as translators.

#### Twitter Scripts
Twitter_Word_Vectors is a word to vector embedding algorithm trained using CBOW on the same dataset as was used to train the twitter sentiment classifiers.  Once the loss function has been improved it will hopefully be used to generate new classifiers to gain more information from the tweets

Twitter_Sentiment_Analysis pulls a livestream from twitter, currently focused on the keyword set "government shutdown".  It then classifies the tweets based on the party they seem most likely to be addressing, as well as the sentiment expressed in the tweet, either positive or neutral, using a vote classifier.

Graph_Twitter_Data uses the data from Twitter_Sentiment_Analysis to create actively updating graphs representing several factors, such as the sentiment towards both parties over time, as well as the apparent sentiment for each party against the other.  Sentiment between the two parties is classified by counting negative sentement for one party as positive sentiment towards the other
