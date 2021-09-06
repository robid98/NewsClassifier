# This class is used for processing the text from news
# Ex : remove punctation, remove stop words etc ..
""" For this we will use nltk library"""
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re


def convert_tag(nltk_pos_tag):
    if nltk_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_pos_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_words(sentence):
    lemmatized_sentence = []
    lemmatizer = WordNetLemmatizer()  # from Nltk
    # POS tag for every word
    pos_tag_words = nltk.pos_tag(nltk.word_tokenize(sentence))

    for word, tag in pos_tag_words:
        converted_tag = convert_tag(tag)
        if converted_tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, converted_tag))

    return ' '.join(lemmatized_sentence)


class ProccessNews:
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.punctuation_exclude = set(string.punctuation)

    def remove_stop_words(self, sentence):
        sentence_words = word_tokenize(sentence)
        sentence_without_stopwords = []
        relevant_words = []
        for word in sentence_words:
            if word not in self.stopwords:
                sentence_without_stopwords.append(word)

        for word in sentence_without_stopwords:
            if len(word) >= 3:
                relevant_words.append(word)

        return ' '.join(relevant_words)

    def remove_punctuation(self, sentence):
        sentence = sentence.lower()
        for pct in self.punctuation_exclude:
            sentence = sentence.replace(pct, " ")

        sentence = re.sub(r'\d+', ' ', sentence)
        sentence = re.sub(r'[^\w]', ' ', " ".join(re.split("\s+", sentence, flags=re.UNICODE)))

        return sentence
