import tweepy
import pickle
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

# Get credentials from local file
with open('secrets.txt') as secrets:
    consumer_key, consumer_secret, \
    access_token, access_secret = map(lambda x: x.strip(), secrets.readlines())

# Open pre-downloaded data if present
try:
    f = open("tweetdump.pkl","rb")
    searched_tweets = pickle.load(f)
    f.close()
# else get from twitter api and save
except FileNotFoundError:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    query = 'novartis'
    max_tweets = 100
    searched_tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]
    # save for next time
    f = open("tweetdump.pkl","wb")
    pickle.dump(searched_tweets,f)
    f.close()

# concatenate tweets
alltext = ''.join([tweet.text for tweet in searched_tweets])

# naive bayes on the text blob
theblob = TextBlob(alltext,analyzer=NaiveBayesAnalyzer())
# sentiment analyser on the blob
print(theblob.sentiment)

