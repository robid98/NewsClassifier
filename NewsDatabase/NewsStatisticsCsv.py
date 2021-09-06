# In this class we will make some functions to help us with some dataset statistics
# For example, how many times a word apppear in the dataset, etc..
# We will display this statistics in some Graphs using matplotlib
import enchant
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import words

from NewsDatabase.StoreMongoDb import StoreMongoDb


class NewsStatisticsCsv:
    def __init__(self):
        self.MongoDb = StoreMongoDb()
        self.frequencyDict = {}
        self.frequencyList = []
        self.frequencyTrueAndFalseNewsDict = {'True': 0, 'False': 0}
        self.frequencyTrueAndFalseNewsList = []
        self.plt = plt

    def reduce_vocabulary_size(self):
        """ Feature extraction , only relevant words """
        """Need to reduce the vocabulary size because is too big > 105000 words"""
        # Iterate over all keys and check if is a valid word in english
        delete_keys_set = set()
        nltk_words = set(words.words())
        enchant_dict = enchant.Dict("en_US")
        for key in self.frequencyDict:
            if self.frequencyDict[key] < 50:
                delete_keys_set.add(key)
            if enchant_dict.check(key) is not True and key not in nltk_words:
                delete_keys_set.add(key)

        # self.unique_words will contain only the relevant words from Database
        for delete_key in delete_keys_set:
            del (self.frequencyDict[delete_key])

        # vocabulary = open("./datasets/vocabulary.txt", "a")
        # for key in self.frequencyDict:
        #    vocabulary.write(key + "\n")
        # vocabulary.close()

        # print(len(self.frequencyDict))

    def calculate_frequency(self):
        news_database = self.MongoDb.collection.find()

        counter = 0
        for news in news_database:
            word_splitted = news['resultedText'].split(" ")

            if news['classified'] == 'False':
                self.frequencyTrueAndFalseNewsDict['False'] = self.frequencyTrueAndFalseNewsDict['False'] + 1
            else:
                self.frequencyTrueAndFalseNewsDict['True'] = self.frequencyTrueAndFalseNewsDict['True'] + 1

            # iterate every splited sentence and add every word in dictionary
            for word in word_splitted:
                if word in self.frequencyDict:
                    self.frequencyDict[word] = self.frequencyDict[word] + 1
                else:
                    self.frequencyDict[word] = 1

        del (self.frequencyDict[""])

        self.reduce_vocabulary_size()

    def word_frequency_dataset(self):

        self.calculate_frequency()

        self.frequencyList = sorted(self.frequencyDict.items(), key=lambda x: x[1]).copy()
        self.frequencyTrueAndFalseNewsList = sorted(self.frequencyTrueAndFalseNewsDict.items(),
                                                    key=lambda x: x[1]).copy()
        # print(self.frequencyTrueAndFalseNewsList)
        # print(self.frequencyList)

        top_words = pd.DataFrame(self.frequencyList[-30:], columns=['News words', 'Count'])
        low_words = pd.DataFrame(self.frequencyList[0:30], columns=['News words', 'Count'])
        true_and_false = pd.DataFrame(self.frequencyTrueAndFalseNewsList, columns=['News State', 'Number'])

        # Plot for first 30 High Frequency Words
        figure, axs = self.plt.subplots(figsize=(9, 9))
        top_words.plot.barh(x='News words',
                            y='Count',
                            ax=axs,
                            color="yellow")
        axs.set_title("High Words frequency in News dataset")

        # Plot for first 30 Low Frequency Words
        self.plt.figure(1)
        figure, axs = self.plt.subplots(figsize=(9, 9))
        low_words.plot.barh(x='News words',
                            y='Count',
                            ax=axs,
                            color="yellow")
        axs.set_title("Low Words frequency in News dataset")

        # Plot for News Statements : True / False
        self.plt.figure(2)
        figure, axs = self.plt.subplots(figsize=(5, 5))
        true_and_false.plot.barh(x='News State',
                                 y='Number',
                                 ax=axs,
                                 color="yellow")
        axs.set_title("News statements in News dataset")

    def show_plt(self):
        self.plt.show()


if __name__ == "__main__":
    newsStatistics = NewsStatisticsCsv()
    newsStatistics.word_frequency_dataset()
    newsStatistics.show_plt()
