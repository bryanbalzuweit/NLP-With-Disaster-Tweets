# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:30:22 2020

@author: Bryan Balzuweit
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from textatistic import Textatistic
import en_core_web_sm
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import re
from nltk.tokenize import word_tokenize


#Create necessary functions
def word_count(string):
    words = string.split()
    return len(words)

def avg_word_length(x):
    words = x.split()
    word_lengths = [len(word) for word in words]
    avg_word_length = sum(word_lengths)/len(words)
    return(avg_word_length)
    
def hashtag_count(string):
    words = string.split()
    hashtags = [word for word in words if word.startswith('#')]
    return len(hashtags)

def mention_count(string):
    words = string.split()
    mentions = [word for word in words if word.startswith('@')]
    return len(mentions)  

def num_sentences(string):
    words = string.split()     
    periods = [word for word in words if word.endswith('.')]
    return len(periods)  

def num_starts_with_upper(string):
    final_data = re.findall("[A-Z]+[a-z]+", string)
    return len(final_data)

def num_all_caps(string):
    final_data = re.findall("[A-Z]+[A-Z]+", string)
    return len(final_data)

def read_score(string):
    # Compute the readability scores
    readability_scores = Textatistic(string).scores
    # Calculate flesch reading ease score
    try:
        flesch = readability_scores['gunningfog_score'] 
    except:
        flesch = 0
    return flesch


nlp = en_core_web_sm.load()


# Returns number of other nouns
def nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of other nouns
    return pos.count("NOUN")

def prop_nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of other nouns
    return pos.count("PROPN")

def verbs(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of other nouns
    return pos.count("VERB")

def clean_location(text, model=nlp):
  	# Create doc object
    doc = model(text)
    
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    joined = ' '.join(lemmas)
    doc = model(joined)
    # Generate list of GPE
    area = [token.text for token in doc if  token.ent_type_ == "GPE"]
    area51 = ""
    if(len(area)==0):
        area51 = ""
    else:
        area51 = area[0]
    
    # Return number of other nouns
    return area51

def has_location(text, model=nlp):
    doc = model(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
    if(len(locations) > 0):
        return 1
    else:
        return 0
    
def to_low(text):
    lowered = text.lower()
    return lowered

def to_tok(text):
    words = word_tokenize(text)
    return words
    

def to_lem(text):
    doc = nlp(text)
    # Extract the lemma for each token and join
    return " ".join([token.lemma_ for token in doc])


#End functions

#Locations of files
location = r'G:\Kaggle\RealOrFake'
train = pd.read_csv(location + r'/train.csv')
test = pd.read_csv(location + r'/test.csv')
sample_submission = pd.read_csv(location + r'/sample_submission.csv')

#Creating the training and test dataframes
y_train = train['target']
X_train = train.copy().drop('target', axis = 1)
X_test = test.copy()

#Filling na values
X_train['keyword'].fillna(999, inplace=True)
X_test['keyword'].fillna(999, inplace=True)

#Feature Engineering
X_train['num_chars'] = X_train['text'].apply(len)
X_test['num_chars'] = X_test['text'].apply(len)

X_train['num_words'] = X_train['text'].apply(word_count)
X_test['num_words'] = X_test['text'].apply(word_count)

X_train['avg_word_length'] = X_train['text'].apply(avg_word_length)
X_test['avg_word_length'] = X_test['text'].apply(avg_word_length)

X_train['hashtag_count'] = X_train['text'].apply(hashtag_count)
X_test['hashtag_count'] = X_test['text'].apply(hashtag_count)

X_train['mention_count'] = X_train['text'].apply(mention_count)
X_test['mention_count'] = X_test['text'].apply(mention_count)

X_train['sentence_count'] = X_train['text'].apply(num_sentences)
X_test['sentence_count'] = X_test['text'].apply(num_sentences)

X_train['num_starts_with_upper'] = X_train['text'].apply(num_starts_with_upper)
X_test['num_starts_with_upper'] = X_test['text'].apply(num_starts_with_upper)

X_train['num_all_caps'] = X_train['text'].apply(num_all_caps)
X_test['num_all_caps'] = X_test['text'].apply(num_all_caps)

X_train['num_nouns'] = X_train['text'].apply(nouns)
X_test['num_nouns'] = X_test['text'].apply(nouns)

X_train['num_propnouns'] = X_train['text'].apply(prop_nouns)
X_test['num_propnouns'] = X_test['text'].apply(prop_nouns)

X_train['num_verbs'] = X_train['text'].apply(verbs)
X_test['num_verbs'] = X_test['text'].apply(verbs)

X_train['has_location'] = X_train['text'].apply(has_location)
X_test['has_location'] = X_test['text'].apply(has_location)

#Feature encoding
X_train["keyword"] = X_train["keyword"].astype('category')
X_train.dtypes
X_train["keyword_label"] = X_train["keyword"].cat.codes
X_train.drop('keyword', inplace=True, axis=1)

X_test["keyword"] = X_test["keyword"].astype('category')
X_test.dtypes
X_test["keyword_label"] = X_test["keyword"].cat.codes
X_test.drop('keyword', inplace=True, axis=1)

"""
Feature selection is commented out because adding any of these features actually worsens the f1 score.
Below I use only the count vectorizer as the input variables.
"""
##Feature Selection
##Add the target for correlation
#X_train["target"] = y_train
##Correlate the variables
#cor = X_train.corr()
##Correlation with output variable
#cor_target = abs(cor["target"])
#cor_target.drop(["id", "target"], inplace=True)
##Selecting highly correlated features
#relevant_features = cor_target[cor_target>0.18]
#
#X_train.drop("target", inplace=True, axis=1)
#features = list(relevant_features.index)

#Prepare the text column for vectorizer
#lowercase
X_train['text'] = X_train['text'].apply(to_low)
X_test['text'] = X_test['text'].apply(to_low)

#Lemmatize
X_train['text'] = X_train['text'].apply(to_lem)
X_test['text'] = X_test['text'].apply(to_lem)



# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(strip_accents="ascii", lowercase=True, stop_words="english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train['text'])
count_train_array = count_train.toarray()
df = pd.DataFrame(count_train_array)
#df1 = pd.concat([X_train[features], df], axis=1)


# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test['text'])
count_test_array = count_test.toarray()
df3 = pd.DataFrame(count_test_array)
#df4 = pd.concat([X_test[features], df3], axis=1)


#Create a df that will save the machine learning algorithm name and its respective f1 score
alg_df = pd.DataFrame(columns=["algorithm", "f1_score"])

#This method will run a 5 fold cross validation and save the best f1 score in the alg_df dataframe
def testAlg(alg, name):
    scores = cross_val_score(alg, df, y_train, cv=5, scoring='f1_weighted')
    f1 = max(scores)
    return {"algorithm" : name , "f1_score" : f1}

#Try Naive Bayes classifier
nb_classifier = MultinomialNB()
#Obtain score and append it to the df
alg_df = alg_df.append(testAlg(nb_classifier, "MultinomialNB"), ignore_index=True)

#Try linearSVC classifier
linearsvc_classifier = LinearSVC()
#Obtain score and append it to the df
alg_df = alg_df.append(testAlg(linearsvc_classifier, "LinearSVC"), ignore_index=True)

#Try tree classifier
tree_classifier = tree.DecisionTreeClassifier()
#Obtain score and append it to the df
alg_df = alg_df.append(testAlg(tree_classifier, "DecisionTreeClassifier"), ignore_index=True)

#Try rf classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
#Obtain score and append it to the df
alg_df = alg_df.append(testAlg(rf_classifier, "RandomForestClassifier"), ignore_index=True)

#Try xgb classifier
xgb_classifier = XGBClassifier()
#Obtain score and append it to the df
alg_df = alg_df.append(testAlg(xgb_classifier, "XGBClassifier"), ignore_index=True)


#Pick the best classifier
best_alg = list(alg_df.loc[alg_df['f1_score'].idxmax()])
best_alg = best_alg[0]

#Need to add parenthesis so the eval function can execute the string
optimal_classifier = str(best_alg + "()")
#Instantiate the classifier
optimal_classifier = eval(optimal_classifier)
#Fit the classifier to the training data
optimal_classifier.fit(df, y_train)
#Create the predicted tags: pred
pred = optimal_classifier.predict(df3)
#Write the olutput to the file
sample_submission['target'] = pred
sample_submission.to_csv(location + r'\outputs\kaggle_nlp.csv', index=False)
