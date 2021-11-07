import tweepy
from dotenv import load_dotenv
import os
import re
import nltk 
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
from csv import DictReader
import pickle
import pandas as pd


def apiAuthenticaion():
    load_dotenv()
    CONSUMER_KEY = os.getenv('CONSUMER_KEY')
    CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
    ACCESS_KEY = os.getenv('ACCESS_KEY')
    ACCESS_SECRET = os.getenv('ACCESS_SECRET')
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api
def reorder():
    df = pd.read_csv('16mil.csv',encoding = "ISO-8859-1")
    df.columns=['label','id','date','query','name','text']
    df.drop(columns=['query','name','date'],inplace=True)
    df['label'] = df['label'].replace(['0'],'negative')
    df_reorder = df[["text", "label", "id"]]
    df_reorder.to_csv('16milcleaned.csv', index=False,quotechar='"',doublequote=True)
def getTrainingData():
    file="16milcleaned.csv"
    with open(file, 'r',encoding="utf-8") as read_obj:
        dict_reader = DictReader(read_obj)
        trainingData = list(dict_reader)
        return trainingData
def fetchtweets(api,keywords):
    try:
        tweets = tweepy.Cursor(api.search_tweets,q=keywords,lang="en").items(100)
        x=[{"text":tweet.text,"label":None} for tweet in tweets]
        print("Fetched " + str(tweets.num_tweets)+ " tweets for the terms " + str(keywords))
        return x

    except:
        print("Unfortunately, something went wrong..")
        return None
class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]
def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, label) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features 




def run():
    #standard tweepy api initialization
    api=apiAuthenticaion()
    #fetching the input of the keyword you would like its sentimnent analyzed
    keyword=input("Enter Your Keyword \n")
    #creating a list of dictionaries for every tweet fetched, containing the text and it's label being none 
    testDataSet=fetchtweets(api,keyword)
    #cleaning the provided dataset
    reorder()
    #fetching the cleaned data set
    trainingData=getTrainingData()
    #initializing an object from our processing class
    tweetProcessor = PreProcessTweets()
    #processing and tokenizing both our test and training sets
    preprocessedTestSet = tweetProcessor.processTweets(testDataSet)
    preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
    #creating a vocabulary and frequnecy feature list from our training set
    word_features = buildVocabulary(preprocessedTrainingSet)
    #apply the features and train the model, gonna take a while
    trainingFeatures = nltk.classify.apply_features(extract_features, preprocessedTrainingSet)
    NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    #using the classifer to iterate on the the test set provided 
    NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
    if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
        print("Overall Positive Sentiment")
        print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
    else: 
        print("Overall Negative Sentiment")
        print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")

    #this is optional, if you choose to pickle the classifier to save training time and simply cut to the classification, open the file and use pickle.load to load it once again
    pickle_out = open("NBayesClassifier.pickle","wb")
    pickle.dump(NBayesClassifier, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    run()