import pandas
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import requests
import pickle
import string

debug = 1

text, resolution, problem = [], [], []

print("Loading")
wfile = open('tickets.pkl','rb')
text=pickle.load(wfile)
resolution=pickle.load(wfile)
problem=pickle.load(wfile)
wfile.close()

for i,t in enumerate(text):
    text[i] = ' '.join(t.split())
    
print("Dataframe")
# create a dataframe using descriptions and resolutions/problems
trainDF = pandas.DataFrame()
trainDF['text'] = text
trainDF['label'] = resolution # infer these first

print("Spltting")
# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

print("Labelling")
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print("Fitting")
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(2,2))
#count_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(2,4))
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# Naive Bayes Classifier on the count vectors
classifier = naive_bayes.MultinomialNB()

# fit the training dataset on the classifier
classifier.fit(xtrain_count, train_y)

# predict the labels on validation dataset
predictions = classifier.predict(xvalid_count)
accuracy = metrics.accuracy_score(predictions, valid_y)

print("Correctly predicted", accuracy*100,'%')
vxl = list(valid_x)
d = open('results.csv','w',encoding='UTF-8')
d.write("Actual,Predicted\n")
for i in range(len(valid_y)):
    x = pandas.DataFrame()
    x['valid'] = (valid_y[i],predictions[i])
    try:
        e = encoder.inverse_transform(x)
    except ValueError:
        print("Label not in training set")
    d.write(e[0]+','+e[1]+',"'+vxl[i]+'"\n')
d.close()        

