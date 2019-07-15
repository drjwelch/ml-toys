import re
import nltk
from nltk import FreqDist
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

class KeywordFinder:

    IGNORES = None
    
    def __init__(self, stops=None):
        KeywordFinder.IGNORES = set(stopwords.words('english'))
        KeywordFinder.IGNORES |= set(stops)
        self.model = None
        self.z = 0      # normalising constant for model
        self.emodel, self.emodelz = EnglishModel().get_model()

    def _toklem(self, corpus):
        t = TweetTokenizer()
        tokens = t.tokenize(corpus.lower())
        tokens = [token for token in tokens if re.match("^[a-zA-Z_]*$", token)]
        #print('tokens:',tokens)
        # extract only nouns - tokens became temp for testing; x[0] -> x
        #temp = [x for x in nltk.pos_tag(tokens) if x[1]=='NN']
        #print('POS:',nltk.pos_tag(tokens))
        #
        # TODO: remove adverbs using POS tagging; testing showed they are distracting
        #
        l = WordNetLemmatizer()
        tokens = [l.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if len(token)>1 ]
        print('lemmas:',tokens)
        tokens = [token for token in tokens if token not in KeywordFinder.IGNORES]
        return tokens

    def learn(self, corpus):
        self.model = dict(nltk.FreqDist(self._toklem(corpus)))
        self.z = sum([self.model[word] for word in self.model.keys()])

    def _get_token_prob(self, token, unseen, model, z):
        if token not in model:
            # unseen words set to half the smallest prob
            return unseen/z
        else:
            return model[token]/z
        
    def get_keywords(self, text, n=3, unseen=0.5, probs=True):
        if not self.model: raise ModelUndefined
        tokens = set(self._toklem(text))
        print("candidates:",tokens)
        # the core of this model is that keywords are common
        # English words but that are rare in the training set
        ranked = []
        for t in tokens:
            token_model_prob = self._get_token_prob(t, unseen, self.model, self.z)
            token_english_prob = self._get_token_prob(t, 1000, self.emodel, self.emodelz)
            print(t,int(token_model_prob*1000),int(token_english_prob*100000))
            token_keywordness = token_english_prob / token_model_prob
            ranked.append([t, token_keywordness])
        ranked = sorted(ranked, key=lambda x:x[1])
        if len(ranked)<n: raise TooFewWords(ranked)
        if probs:
            return ranked[:n]
        else:
            return [ranked[i][0] for i in range(n)]

class ModelUndefined(Exception):
    pass

class TooFewWords(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class EnglishModel:

    # trains a model on the English 'brown' text
    def __init__(self):
        self.model = dict(FreqDist(i.lower() for i in brown.words())) # English text generally
        self.z = sum([self.model[x] for x in self.model.keys()])

    def get_model(self):
        return self.model, self.z

    # DEPRECATED
    # request prob of word in that corpus
    def get_prob(self, word):
        return self.probs[word] if word in self.probs else 0

