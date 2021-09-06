# Build a KNN Classifier algorithm from scratch
import math
import os
import pickle

from NewsDatabase.NewsStatisticsScrapedNews import NewsStatisticsScrapedNews
import pandas as pd

from NewsDatabase.ProcessNews import ProccessNews, lemmatize_words

BASE = os.path.dirname(os.path.abspath(__file__))


class KNNAlgorithm:

    def __init__(self):
        # Unique categories and unique words in all Documents
        self.news_statistics_scraped_obj = NewsStatisticsScrapedNews()

        # mongodb scraped news in csv
        self.scraped_news_csv = pd.read_csv(os.path.join(BASE, "datasets_scraped"
                                                               "\dataset_news_classifier_scraped_shuffled.csv"))

        # mongodb scraped news in csv test
        self.scraped_news_test_csv = pd.read_csv(os.path.join(BASE, "datasets_scraped"
                                                                    "\dataset_news_classifier_scraped_test.csv"))

        # number of total articles
        self.number_of_articles = len(self.scraped_news_csv)

        # every word in dataset occurence percentage
        self.word_percentage_occurence = {}

        # build dictionary to help KNN process faster
        self.build_knn_dictionary = {}

        self.knn_vocabulary_only = {}

        self.process_news_object = ProccessNews()

    def build_word_percentage_occurence(self):
        self.news_statistics_scraped_obj.load_frequency_dict()

        for word in self.news_statistics_scraped_obj.word_frequency_dict:
            self.word_percentage_occurence[word] = 0

        for times, unique_word in enumerate(self.news_statistics_scraped_obj.word_frequency_dict):
            number_of_word_in_articles = 0
            for index, content, category in zip(self.scraped_news_csv['id'],
                                                self.scraped_news_csv['news_content'],
                                                self.scraped_news_csv['category']):
                if unique_word in content:
                    number_of_word_in_articles += 1
            self.word_percentage_occurence[unique_word] = number_of_word_in_articles / self.number_of_articles
            print(times)

        with open(os.path.join(BASE, "mlclassifier_scraped\knnwordpercentage.pickle"), 'wb') as f:
            pickle.dump(self.word_percentage_occurence, f, pickle.HIGHEST_PROTOCOL)

    def load_word_percentage(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\knnwordpercentage.pickle"), 'rb') as handle:
            self.word_percentage_occurence = pickle.load(handle)

    def build_knn_structure(self):
        self.load_word_percentage()
        self.build_knn_dictionary = {}
        for index, content, category in zip(self.scraped_news_csv['id'],
                                            self.scraped_news_csv['news_content'],
                                            self.scraped_news_csv['category']):
            self.build_knn_dictionary[index] = {}
            content_splitted = content.split(" ")
            for word in content_splitted:
                count_occurence_word = content_splitted.count(word)
                if index in self.build_knn_dictionary:
                    self.build_knn_dictionary[index][word] = count_occurence_word

        for index in self.build_knn_dictionary:
            self.knn_vocabulary_only[index] = {}
            for word in self.build_knn_dictionary[index]:
                if word in self.word_percentage_occurence:
                    value = self.build_knn_dictionary[index][word] * math.log10(
                        1 / self.word_percentage_occurence[word])
                    self.knn_vocabulary_only[index][word] = value

        with open(os.path.join(BASE, "mlclassifier_scraped\knn_dictionary_help.pickle"), 'wb') as f:
            pickle.dump(self.knn_vocabulary_only, f, pickle.HIGHEST_PROTOCOL)

    def load_knn_dictionary(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\knn_dictionary_help.pickle"), 'rb') as handle:
            self.knn_vocabulary_only = pickle.load(handle)

    def knn_algorithm(self, input_sentence, k):
        self.load_knn_dictionary()
        # Process sentence : lemmatize / remove stop words, punctuation, numbers ...
        sentence = lemmatize_words(self.process_news_object.remove_stop_words((
            self.process_news_object.remove_punctuation(input_sentence))))

        input_splitted = sentence.split(" ")

        dict_sentences_score = {}
        counter = 0
        for index in self.knn_vocabulary_only:
            sentence_score = 0
            for word in input_splitted:
                if word in self.knn_vocabulary_only[index]:
                    sentence_score = sentence_score + self.knn_vocabulary_only[index][word]
            dict_sentences_score[index] = sentence_score
            counter += 1

        dict_sorted = sorted(dict_sentences_score.items(), key=lambda x: x[1], reverse=True)

        frequency_first_k_category = {}
        for pair in dict_sorted[0:k]:
            row = self.scraped_news_csv.loc[self.scraped_news_csv['id'] == pair[0]]
            for r in row['category']:
                if r in frequency_first_k_category:
                    frequency_first_k_category[r] += 1
                else:
                    frequency_first_k_category[r] = 1

        # best category
        max_category_freq = -1
        save_category = ""
        for category in frequency_first_k_category:
            if frequency_first_k_category[category] > max_category_freq:
                max_category_freq = frequency_first_k_category[category]
                save_category = category

        return save_category

    def test_model(self):
        self.news_statistics_scraped_obj.load_frequency_dict()
        # test the model with the articles from 2018
        classified_correct = 0
        index = 0
        test_size = int(len(self.scraped_news_test_csv) * 0.005)
        print(test_size)
        for content, category in zip(self.scraped_news_test_csv['news_content'],
                                     self.scraped_news_test_csv['category']):
            if category in self.news_statistics_scraped_obj.category_frequency_dict:
                index += 1
                classified_category = self.knn_algorithm(content, 5)
                if classified_category == category:
                    classified_correct += 1
            if index == test_size:
                break
            print(index)

        print("Classifier accuray: " + str(classified_correct / test_size))

# knn = KNNAlgorithm()
# knn.build_word_percentage_occurence()
# knn.test_model()
