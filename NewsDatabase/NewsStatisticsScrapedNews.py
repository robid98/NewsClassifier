""" In this python file we will make a class that will contain multiple functions to show different
statistics about the scraped articles with New York Times Api """
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import enchant
from nltk.corpus import words

from NewsDatabase.StoreMongoDb import StoreMongoDb

BASE = os.path.dirname(os.path.abspath(__file__))


class NewsStatisticsScrapedNews:
    def __init__(self):
        self.MongoDb = StoreMongoDb()
        self.word_frequency_dict = {}
        self.category_frequency_dict = {}
        self.sentences_words_length = {}

    def scraped_news_frequency(self):
        article_database = self.MongoDb.collection_scraped.find()

        article_id = 0
        for article in article_database:
            article_words = 0
            article_splitted = article['content'].split(" ")

            # Content
            for word in article_splitted:
                article_words = article_words + 1
                if word in self.word_frequency_dict:
                    self.word_frequency_dict[word] = self.word_frequency_dict[word] + 1
                else:
                    self.word_frequency_dict[word] = 1

            # Category
            category = article['category']
            if category in self.category_frequency_dict:
                self.category_frequency_dict[category] = self.category_frequency_dict[category] + 1
            else:
                self.category_frequency_dict[category] = 1

            self.sentences_words_length[str(article_id)] = article_words

            article_id = article_id + 1

        # Reduce vocabulary set, too much words
        delete_keys_set = set()
        nltk_words = set(words.words())
        enchant_dict = enchant.Dict("en_US")
        for word in self.word_frequency_dict:
            if self.word_frequency_dict[word] < 50:
                delete_keys_set.add(word)
            if enchant_dict.check(word) is not True and word not in nltk_words:
                delete_keys_set.add(word)

        for delete_key in delete_keys_set:
            del (self.word_frequency_dict[delete_key])

        # with open('./frequency_scraped/word_frequency_scraped.pickle', 'wb') as f:
        #    pickle.dump(self.word_frequency_dict, f, pickle.HIGHEST_PROTOCOL)

        # with open('./frequency_scraped/category_frequency_scraped.pickle', 'wb') as f:
        #    pickle.dump(self.category_frequency_dict, f, pickle.HIGHEST_PROTOCOL)

        # with open('./frequency_scraped/article_word_length.pickle', 'wb') as f:
        #    pickle.dump(self.sentences_words_length, f, pickle.HIGHEST_PROTOCOL)

    def load_frequency_dict(self):
        with open(os.path.join(BASE, "frequency_scraped\word_frequency_scraped.pickle"), 'rb') as handle:
            self.word_frequency_dict = pickle.load(handle)
        with open(os.path.join(BASE, "frequency_scraped\category_frequency_scraped.pickle"), 'rb') as handle:
            self.category_frequency_dict = pickle.load(handle)
        with open(os.path.join(BASE, "frequency_scraped\\article_word_length.pickle"), 'rb') as handle:
            self.sentences_words_length = pickle.load(handle)

    def build_plots(self):
        self.load_frequency_dict()

        word_frequency_list = sorted(self.word_frequency_dict.items(), key=lambda x: x[1])
        category_frequency_list = sorted(self.category_frequency_dict.items(), key=lambda x: x[1])
        article_word_length = sorted(self.sentences_words_length.items(), key=lambda x: x[1])

        matplotlib.rcParams.update({'font.size': 8})
        plt.figure(figsize=(16, 6))
        plt.bar(*zip(*category_frequency_list[-20:]), color="yellow")
        plt.xticks(rotation=90)
        plt.title("Category frequency in scraped News")

        plt.figure(1)
        plt.figure(figsize=(10, 6))
        plt.bar(*zip(*word_frequency_list[-20:]), color="blue")
        plt.xticks(rotation=90)
        plt.title("Word frequency in scraped News")

        plt.figure(2)
        plt.figure(figsize=(10, 6))
        plt.bar(*zip(*article_word_length[-20:]), color="purple")
        plt.title("Words count in every news ( OX: Article Id, OY: Number of words )")

        plt.show()


if __name__ == "__main__":
    scraped_news_obj = NewsStatisticsScrapedNews()
    scraped_news_obj.build_plots()
