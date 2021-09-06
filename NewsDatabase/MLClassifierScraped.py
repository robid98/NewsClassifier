# Implementation from scratch of the Multinomial Naive Bayes with TF-IDF instead of Bag of Words
import csv
import os
import pickle
from decimal import Decimal

from NewsDatabase.NewsStatisticsScrapedNews import NewsStatisticsScrapedNews
from NewsDatabase.ProcessNews import ProccessNews, lemmatize_words
from NewsDatabase.StoreMongoDb import StoreMongoDb
import pandas as pd
import numpy as np
import operator

BASE = os.path.dirname(os.path.abspath(__file__))


def shuffle_processed_csv(path):
    # In this function we will shuffle the processed csv and save it in another csv, for further processing
    processed_csv = pd.read_csv(path)
    processed_csv = processed_csv.sample(frac=1).reset_index(drop=True)

    with open(os.path.join(BASE, "datasets_scraped\dataset_news_classifier_scraped_shuffled.csv"), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "news_content", "category"])
        for id_news, text_news, classification in zip(processed_csv['id'], processed_csv['news_content'],
                                                      processed_csv['category']):
            csv_row = [id_news, text_news, classification]
            writer.writerow(csv_row)


class MLClassifierScraped:
    def __init__(self):
        self.mongodb_object = StoreMongoDb()

        # in this dictionary we will save the values resulted from TF
        self.store_tf = {}
        # in this dictionary we will save the values resulted from IDF
        self.store_idf = {}
        # in this dictionary we will save the result obtained from TF-IDF
        self.store_tf_idf = {}

        # Unique categories and unique words in all Documents
        self.news_statistics_scraped_obj = NewsStatisticsScrapedNews()

        # sum(tf_idf for all words in that category)
        self.sum_tf_idf_all_words_each_category = {}

        # sum(tf_idf for a word in a specific category)
        self.sum_tf_idf_word_in_specific_category = {}

        # bag of words - number of a word occurence in every category he is presented
        self.bag_of_words = {}

        # how many words in each category
        self.each_category_vocabulary = {}

        # build probabilities with Multinomial Naive Bayes and Bag of Words
        self.build_probabilities_bag_of_words = {}

        # probabilities P(word | category) => each word , in each category that appear
        self.word_probabilities = {}

        # Categories probabilities : Priors
        self.categories_probabilities = {}

        # mongodb scraped news in csv
        self.scraped_news_csv = pd.read_csv(os.path.join(BASE, "datasets_scraped"
                                                               "\dataset_news_classifier_scraped_shuffled.csv"))

        # mongodb scraped news in csv test
        self.scraped_news_test_csv = pd.read_csv(os.path.join(BASE, "datasets_scraped"
                                                                    "\dataset_news_classifier_scraped_test.csv"))

        self.process_news_object = ProccessNews()

    def load_unique_categories_and_words(self):
        self.news_statistics_scraped_obj.load_frequency_dict()

    def calculate_tf(self):
        self.load_unique_categories_and_words()

        # Iterate every sentence from scraped_news_csv  and count frequency
        for unique_word in self.news_statistics_scraped_obj.word_frequency_dict:
            self.store_tf[unique_word] = []

        for times, unique_word in enumerate(self.news_statistics_scraped_obj.word_frequency_dict):
            for index, content, category in zip(self.scraped_news_csv['id'],
                                                self.scraped_news_csv['news_content'],
                                                self.scraped_news_csv['category']):
                sentence_splitted = content.split(" ")
                count_unique_word = sentence_splitted.count(unique_word)
                if count_unique_word > 0:
                    # self.store_tf[unique_word].append((index, count_unique_word, category))
                    self.store_tf[unique_word].append((index, 1 + np.log(count_unique_word), category))
            print(times)

        with open(os.path.join(BASE, "mlclassifier_scraped\\tf.pickle"), 'wb') as f:
            pickle.dump(self.store_tf, f, pickle.HIGHEST_PROTOCOL)

    def load_tf(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\\tf.pickle"), 'rb') as handle:
            self.store_tf = pickle.load(handle)

    def build_bag_of_words_from_tf(self):
        # self.load_unique_categories_and_words()
        # self.load_categories_probabilities()
        # self.load_tf()

        for unique_word in self.store_tf:
            self.bag_of_words[unique_word] = {}

        for unique_word in self.store_tf:
            for group in self.store_tf[unique_word]:
                if group[2] not in self.bag_of_words[unique_word]:
                    self.bag_of_words[unique_word][group[2]] = group[1]
                else:
                    self.bag_of_words[unique_word][group[2]] += group[1]

        # with open('./mlclassifier_scraped/bag_of_words.pickle', 'wb') as f:
        #    pickle.dump(self.bag_of_words, f, pickle.HIGHEST_PROTOCOL)

        # each category vocabulary
        for unique_category in self.news_statistics_scraped_obj.category_frequency_dict:
            self.each_category_vocabulary[unique_category] = 0

        for unique_category in self.each_category_vocabulary:
            for unique_word in self.bag_of_words:
                if unique_category in self.bag_of_words[unique_word]:
                    self.each_category_vocabulary[unique_category] += self.bag_of_words[unique_word][unique_category]

        # build the probabiltiies
        for unique_word in self.store_tf:
            self.build_probabilities_bag_of_words[unique_word] = {}

        for unique_word in self.build_probabilities_bag_of_words:
            for unique_category in self.news_statistics_scraped_obj.category_frequency_dict:
                # P(unique_word | unique_category)
                total_number_of_words_in_class = self.each_category_vocabulary[unique_category]

                if unique_category in self.bag_of_words[unique_word]:
                    unique_word_in_class = self.bag_of_words[unique_word][unique_category]
                else:
                    unique_word_in_class = 0

                self.build_probabilities_bag_of_words[unique_word][unique_category] = (unique_word_in_class + 1) / (
                        total_number_of_words_in_class + len(self.bag_of_words))

        # with open('./mlclassifier_scraped/word_probabilities_with_bag_of_words.pickle', 'wb') as f:
        #    pickle.dump(self.build_probabilities_bag_of_words, f, pickle.HIGHEST_PROTOCOL)

    def load_bag_of_words(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\\bag_of_words.pickle"), 'rb') as handle:
            self.bag_of_words = pickle.load(handle)

    def calculate_idf(self):
        number_of_documents = len(self.scraped_news_csv)
        self.load_tf()
        for unique_word in self.store_tf:
            self.store_idf[unique_word] = np.log(number_of_documents / len(self.store_tf[unique_word]))

        with open(os.path.join(BASE, "mlclassifier_scraped\idf.pickle"), 'wb') as f:
            pickle.dump(self.store_idf, f, pickle.HIGHEST_PROTOCOL)

    def load_idf(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\idf.pickle"), 'rb') as handle:
            self.store_idf = pickle.load(handle)

    def calculate_tf_idf(self):
        self.load_tf()
        self.load_idf()

        for unique_word in self.store_tf:
            self.store_tf_idf[unique_word] = []

        for unique_word in self.store_tf_idf:
            for pair in self.store_tf[unique_word]:
                if (self.store_idf[unique_word] * pair[1]) > 0:
                    self.store_tf_idf[unique_word].append((pair[0], self.store_idf[unique_word] * pair[1], pair[2]))

        with open(os.path.join(BASE, "mlclassifier_scraped\\tf_idf.pickle"), 'wb') as f:
            pickle.dump(self.store_tf_idf, f, pickle.HIGHEST_PROTOCOL)

    def load_tf_idf(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\\tf_idf.pickle"), 'rb') as handle:
            self.store_tf_idf = pickle.load(handle)

    def calculate_probababilities_every_class(self):
        self.load_unique_categories_and_words()
        self.load_tf_idf()

        for unique_category in self.news_statistics_scraped_obj.category_frequency_dict:
            self.sum_tf_idf_all_words_each_category[unique_category] = 0

        for unique_word in self.store_tf_idf:
            self.sum_tf_idf_word_in_specific_category[unique_word] = {}

        for unique_word in self.store_tf_idf:
            self.word_probabilities[unique_word] = {}

        "P(word|category)=(word_count_in_category + 1)/(total_words_in_category + total_unique_words_in_all_categories)"
        "To use TF-IDF => P(word|category) = sum(tf_idf for word in that category + 1)/(sum(tf_idf for all words in that " \
        "category) + total_unique_words_all_categories"" "
        # self.news_statistics_scraped_obj.category_frequency_dict => all distinct categories

        # sum(tf_idf for all words in that category) => store in a dictionary key : category , value : sum
        for unique_category in self.sum_tf_idf_all_words_each_category:
            for unique_word in self.store_tf_idf:
                for group in self.store_tf_idf[unique_word]:
                    if group[2] == unique_category:
                        self.sum_tf_idf_all_words_each_category[unique_category] += group[1]

        for unique_word in self.sum_tf_idf_word_in_specific_category:
            for group in self.store_tf_idf[unique_word]:
                if group[2] not in self.sum_tf_idf_word_in_specific_category[unique_word]:
                    self.sum_tf_idf_word_in_specific_category[unique_word][group[2]] = group[1]
                else:
                    self.sum_tf_idf_word_in_specific_category[unique_word][group[2]] += group[1]

        # conditional probabilities
        for unique_word in self.word_probabilities:
            for unique_category in self.news_statistics_scraped_obj.category_frequency_dict:
                if unique_category in self.sum_tf_idf_word_in_specific_category[unique_word]:
                    tf_idf_all_words_each_category = self.sum_tf_idf_word_in_specific_category[unique_word][
                        unique_category]
                else:
                    tf_idf_all_words_each_category = 0
                self.word_probabilities[unique_word][unique_category] = (tf_idf_all_words_each_category + 1) / (
                        self.sum_tf_idf_all_words_each_category[unique_category] + len(self.store_tf_idf))

        with open(os.path.join(BASE, "mlclassifier_scraped\word_probabilities.pickle"), 'wb') as f:
            pickle.dump(self.word_probabilities, f, pickle.HIGHEST_PROTOCOL)

    def load_word_probabilities(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\word_probabilities.pickle"), 'rb') as handle:
            self.word_probabilities = pickle.load(handle)

    def load_word_probabilties_with_bag_of_words(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\word_probabilities_with_bag_of_words.pickle"), 'rb') as handle:
            self.build_probabilities_bag_of_words = pickle.load(handle)

    def category_probabilities(self):
        self.load_unique_categories_and_words()

        sum_total_categories_values = 0
        for unique_category in self.news_statistics_scraped_obj.category_frequency_dict:
            self.categories_probabilities[unique_category] = 0
            sum_total_categories_values += self.news_statistics_scraped_obj.category_frequency_dict[unique_category]

        for unique_category in self.categories_probabilities:
            self.categories_probabilities[unique_category] = self.news_statistics_scraped_obj.category_frequency_dict[
                                                                 unique_category] / sum_total_categories_values

        with open(os.path.join(BASE, "mlclassifier_scraped\categories_probabilities.pickle"), 'wb') as f:
            pickle.dump(self.categories_probabilities, f, pickle.HIGHEST_PROTOCOL)

    def load_categories_probabilities(self):
        with open(os.path.join(BASE, "mlclassifier_scraped\categories_probabilities.pickle"), 'rb') as handle:
            self.categories_probabilities = pickle.load(handle)

    def classify_sentence(self, sentence, feature_selection_type):
        # feature_selection_type == 1 -> tf-idf
        # feature_selection_type == 2 -> bag_of_words
        if feature_selection_type == 1:
            self.load_word_probabilities()
        else:
            self.load_word_probabilties_with_bag_of_words()

        self.load_categories_probabilities()

        # Process sentence : lemmatize / remove stop words, punctuation, numbers ...
        sentence = lemmatize_words(self.process_news_object.remove_stop_words((
            self.process_news_object.remove_punctuation(sentence))))

        tokenize_sentence = sentence.split(" ")

        if feature_selection_type == 1:
            used_probabilities = self.word_probabilities
        else:
            used_probabilities = self.build_probabilities_bag_of_words

        for word in tokenize_sentence:
            if word in used_probabilities:
                for unique_category in self.categories_probabilities:
                    self.categories_probabilities[unique_category] = Decimal(
                        self.categories_probabilities[unique_category]) * Decimal(
                        used_probabilities[word][unique_category])
            else:
                # ignore word, not in vocabulary
                pass

        # get the biggest probability from self.categories_probabilities -> return category
        return self.categories_probabilities, max(self.categories_probabilities.items(), key=operator.itemgetter(1))[0]

    def test_model(self):
        self.load_unique_categories_and_words()
        # test the model with the articles from 2018
        classified_correct = 0
        index = 0
        test_size = int(len(self.scraped_news_test_csv) * 0.01)
        print(test_size)
        for content, category in zip(self.scraped_news_test_csv['news_content'],
                                     self.scraped_news_test_csv['category']):
            if category in self.news_statistics_scraped_obj.category_frequency_dict:
                index += 1
                classified_category = self.classify_sentence(content, 1)[1]
                print(classified_category, category, index)
                if classified_category == category:
                    classified_correct += 1
            if index == test_size:
                break

        print("Classifier accuray: " + str(classified_correct / test_size))

mlclassifierscraped = MLClassifierScraped()
# shuffle_processed_csv("./datasets_scraped/dataset_news_classifier_scraped.csv")
# .mlclassifierscraped.calculate_tf()
# mlclassifierscraped.calculate_idf()
# mlclassifierscraped.calculate_tf_idf()
# mlclassifierscraped.calculate_probababilities_every_class()
# mlclassifierscraped.load_word_probabilities()
# mlclassifierscraped.test_model()
# mlclassifierscraped.load_bag_of_words()
