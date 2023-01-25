'''Importing the required packages'''

import tweepy
from tweepy import *
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import numpy as np
import datetime as dt

# Add your API key and secret
consumer_key = 'consumer_key'
consumer_secret = 'consumer_secret'
access_token = 'access_token'
access_token_secret = 'access_token_secret'

# Search for tweets containing the keyword of choice
search_words = 'ArsenalFC'

# number of tweets to search through
no_of_tweets = 100

# creating the sentiment analyser
sid = SentimentIntensityAnalyzer()

'''
Creating the class object to define the pipeline
'''
class TwitterDataPipeline:
    def __init__(self):
        '''Setting the default values in the class'''
        self.auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
        self.auth.set_access_token(access_token,access_token_secret)
        self.api = tweepy.API(self.auth)

    def extract_data(self,search=search_words,tweet_count=10):
        '''extracting data using tweepy, default is set to the search_words variable defined earlier
        default number of tweets searched through is defined before'''
        
        # removing tweets that have been retweeted from the analysis
        query = search_words + " -filter:retweets"
        
        # finding the tweets
        tweets = self.api.search_tweets(q=query, lang="en", tweet_mode='extended',count=tweet_count)
        return tweets

    def transform_data(self, tweets):
        '''transforming the data into the required format'''
        
        # empty list to be appended
        sentiments = []
        for tweet in tweets:
            # the full text of the tweet
            text = tweet.full_text
            
            # sentiment score
            '''
            -1 is very negative, 0 is neutral, 1 is positive - using compound which is a mixure of all scores
            '''
            sentiment = sid.polarity_scores(text)["compound"]
            
            # counting the number of retweets
            retweets = tweet.retweet_count
            
            # number of likes 
            likes = tweet.favorite_count
            
            # the number of followers of the tweeter - used to determine the viewership of the tweet
            followers = tweet.user.followers_count
            
            # time of tweet
            date_time = tweet.created_at
            
            # location of tweet (if available)
            location = tweet.user.location
            
            # appending the empty list with the values  
            sentiments.append([sentiment, retweets, likes, followers,date_time,location])
        return sentiments

    def load_data(self, sentiments):
        '''Writing the results of the transform into a dataframe for analysis'''
        df = pd.DataFrame(sentiments, columns=["Sentiment", "Retweets", "Likes", "Followers","Datetime","Location"])
        return df
      
# Generation of pipeline using the class defined earlier
pipeline = TwitterDataPipeline()
tweets = pipeline.extract_data(tweet_count=no_of_tweets)
sentiments = pipeline.transform_data(tweets)
df = pipeline.load_data(sentiments)

'''
Data analysis
'''
# displaying the info of the dataframe and the dataframe
display(df.info(),df)

# the values were then found to be blank and not null
# changing the blank string values in null values for analysis
df = df.replace('',np.nan)
display(df.info())

# displaying the differing values and their respective counts
print(df['Location'].value_counts(dropna=False))

# too many null values and variation
df = df.drop(columns=['Location'])

# converting the datetime into the specific time
df['Hour'] = df['Datetime'].dt.hour

# converting the datetime into the day of the week
df['Day'] = df['Datetime'].dt.day_name()

# dropping the datetime column
df = df.drop(columns=['Datetime'])
