import re
import string
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CounterVectorizer

# Read Raw data
raw_data = open('datasets/SMSSpamCollection').read()

# print(raw_data[0:500])


parsed_data = raw_data.replace('\t','\n').split('\n')

label_list = parsed_data[0::2]
msg_list = parsed_data[1::2]

print(label_list[0:5])
print(msg_list[1:5])


pd.set_option('display.max_colwidth', 100)

combine_df = pd.DataFrame({
    'label': label_list[:-1],
    'msg' : msg_list
})
print(combine_df.head())



# Exploring data set
# Remove_punctuation
def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


combine_df['msg_clean'] = combine_df['msg'].apply(lambda x : remove_punctuation(x))
print(combine_df.head())




# Tokenization
def tokenize(txt):
    tokens = re.split('\W+',txt)
    return tokens

combine_df['msg_clean_takenized'] = combine_df['msg_clean'].apply(lambda x:tokenize(x.lower()))

print(combine_df.head())



#Remove stop words
stopwords = nltk.corpus.stopwords.word('english')
stopwords[0:10]
def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean
combine_df['msg_no_sw'] = combine_df['msg_clean_takenized'].apply(lambda  x: remove_stopwords(x))
print(combine_df.head())

#Stemming
def stemming(tokenized_text):
    ps = PorterStemmer()
    text = [ps.stem(word) for word in tokenized_text]
    return text

combine_df['msg_stemmed'] = combine_df['msg_no_sw'].apply(lambda x: stemming(x))
combine_df.head()


# Lemmatization
wn = nltk.WordNetLemmatizer()
def lemmatization(token_txt):
    text = [wn.lemmatize(word) for word in token_txt]
    return text

combine_df['msg_lemmatized'] = combine_df['msg_clean_takenized'].apply(lambda x: lemmatization(x))
print(combine_df.head())

# Count vectorization









