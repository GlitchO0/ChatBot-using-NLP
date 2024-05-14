
import nltk
import numpy as np



#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())
def bag_pf_words(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]

    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w,in enumerate(all_words):  #will give us the index and the current word
        if w in tokenized_sentence:
         bag[idx]=1.0
    return bag





# examples -->
#
# sentence=["hello","how","are","you"]
# words=["hi","hello","i","you","bye","thank","cool"]
# bag=bag_pf_words(sentence,words)
# print(bag)


# a="id like to get service"
# print(a)
# a=tokenize(a)
# print(a)
# print('-'*50)
# words=["organize","organizes","organizing"]
# stemmed_words=[stem(w) for w in words]
# print(stemmed_words)
# print('-'*50)
