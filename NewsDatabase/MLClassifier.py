# Implementation from scratch of the Multinomial Naive Bayes with Bag of Words
import csv
import os
from decimal import *

import pandas as pd
import pickle
from NewsDatabase.NewsStatisticsCsv import NewsStatisticsCsv
from NewsDatabase.ProcessNews import ProccessNews, lemmatize_words

BASE = os.path.dirname(os.path.abspath(__file__))


def shuffle_processed_csv(path):
    # In this function we will shuffle the processed csv and save it in another csv, for further processing
    processed_csv = pd.read_csv(path)
    processed_csv = processed_csv.sample(frac=1).reset_index(drop=True)
    with open(os.path.join(BASE, "datasets\dataset_news_classifier_shuffled.csv"), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "news_text", "classification"])
        for id_news, text_news, classification in zip(processed_csv['id'], processed_csv['news_text'],
                                                      processed_csv['classification']):
            csv_row = [id_news, text_news, classification]
            writer.writerow(csv_row)


class MLClassifier:
    def __init__(self):
        self.news_statistics_object = NewsStatisticsCsv()
        self.process_news_object = ProccessNews()
        self.processed_csv = pd.read_csv(os.path.join(BASE, "datasets\dataset_news_classifier_shuffled.csv"))
        self.csv_length = len(self.processed_csv)
        self.training_dataset = []
        self.test_dataset = []

        # Dictionary to build the matrix | how many times each unique word key apppear in every message from the
        # training set
        self.word_key_counter = {}

        # In this dict we will save the conditional probabilities for every word
        self.word_probabilities = {}

    def split_processed_csv(self):
        split_number = int(self.csv_length * 0.8)
        self.training_dataset = self.processed_csv[:split_number].reset_index(drop=True)
        self.test_dataset = self.processed_csv[split_number:].reset_index(drop=True)

        return self.training_dataset, self.test_dataset

    def build_matrix(self):
        self.split_processed_csv()
        """In this function we will transform the sentences in a matrix with numbers
        We will need this because we will work faster on processing the senteces and information"""

        # We will use NewsStatistics Class
        self.news_statistics_object.calculate_frequency()

        # Iterate every sentence from training_dataset and count every word
        for unique_word in self.news_statistics_object.frequencyDict:
            self.word_key_counter[unique_word] = []

        for times, key in enumerate(self.word_key_counter):
            for index, sentence, classification in zip(self.training_dataset['id'], self.training_dataset['news_text'],
                                                       self.training_dataset['classification']):
                sentence_splitted = sentence.split(" ")
                count_key = sentence_splitted.count(key)
                if count_key > 0:
                    self.word_key_counter[key].append((index, count_key, classification))
            print(times)

        with open(os.path.join(BASE, "mlclassifier\\bag_of_words.pickle"), 'wb') as f:
            pickle.dump(self.word_key_counter, f, pickle.HIGHEST_PROTOCOL)

    def load_bag_of_words(self):
        with open(os.path.join(BASE, "mlclassifier\\bag_of_words.pickle"), 'rb') as handle:
            self.word_key_counter = pickle.load(handle)

    def process_matrix(self):
        self.split_processed_csv()
        """In this function i will calculate the probabilities that will help to classify the input"""
        self.load_bag_of_words()

        number_of_true_news = len(self.training_dataset[self.training_dataset['classification'] == 1])
        number_of_fake_news = len(self.training_dataset[self.training_dataset['classification'] == 0])
        total_news = number_of_fake_news + number_of_true_news

        # Vocabulary size
        vocabulary_size = len(self.word_key_counter)

        # Priors
        p_true_news = number_of_true_news / total_news
        p_fake_news = number_of_fake_news / total_news

        # Conditional Probabilities
        for key in self.word_key_counter:
            self.word_probabilities[key] = {}

        # total words occured in true news and fake news
        total_words_true_news = 0
        total_words_fake_news = 0
        for count, unique_word in enumerate(self.word_probabilities):
            for group in self.word_key_counter[unique_word]:
                if group[2] == 1:  # True News
                    total_words_true_news = total_words_true_news + group[1]
                elif group[2] == 0:  # Fake News
                    total_words_fake_news = total_words_fake_news + group[1]

        """Probabilities"""
        for count, unique_word in enumerate(self.word_probabilities):
            total_word_freq_true_news = 0
            total_word_freq_fake_news = 0

            for group in self.word_key_counter[unique_word]:
                if group[2] == 1:  # True news
                    total_word_freq_true_news += group[1]
                elif group[2] == 0:  # Fake news
                    total_word_freq_fake_news += group[1]

            p_true = (total_word_freq_true_news + 1) / (total_words_true_news + vocabulary_size)
            p_false = (total_word_freq_fake_news + 1) / (total_words_fake_news + vocabulary_size)
            self.word_probabilities[unique_word] = {'p_true': p_true, 'p_false': p_false}

        self.word_probabilities['p_true_news'] = p_true_news
        self.word_probabilities['p_fake_news'] = p_fake_news

        with open(os.path.join(BASE, "mlclassifier\words_probabilities.pickle"), 'wb') as f:
            pickle.dump(self.word_probabilities, f, pickle.HIGHEST_PROTOCOL)

    # Use this outside of the class, for better improvment of the time execution
    def load_words_probabilities(self):
        with open(os.path.join(BASE, "mlclassifier\words_probabilities.pickle"), 'rb') as handle:
            self.word_probabilities = pickle.load(handle)

    def classify_sentence(self, sentence):
        self.load_words_probabilities()

        # Process sentence : lemmatize / remove stop words, punctuation, numbers ...
        sentence = lemmatize_words(self.process_news_object.remove_stop_words((
            self.process_news_object.remove_punctuation(sentence))))

        tokenize_sentence = sentence.split(" ")
        p_sentence_is_true = self.word_probabilities['p_true_news']
        p_sentence_is_false = self.word_probabilities['p_fake_news']

        for word in tokenize_sentence:
            if word in self.word_probabilities:
                p_sentence_is_true = Decimal(p_sentence_is_true) * Decimal(self.word_probabilities[word]['p_true'])
                p_sentence_is_false = Decimal(p_sentence_is_false) * Decimal(self.word_probabilities[word]['p_false'])
            else:
                # We need to ignore that word, not in vocabulary of the Classifier
                pass

        if p_sentence_is_true > p_sentence_is_false:
            return 1
        elif p_sentence_is_true < p_sentence_is_false:
            return 0
        else:
            return -1

    def test_mlclassifier(self):
        self.split_processed_csv()

        total_news_test_dataset = len(self.test_dataset)
        total_news_correct_classified = 0
        contor = 0
        for sentence, classified in zip(self.test_dataset['news_text'], self.test_dataset['classification']):
            contor += 1
            return_classified = self.classify_sentence(sentence)
            if return_classified == classified:
                total_news_correct_classified += 1
            print(contor)

        print("Accuray algorithm : " + str(total_news_correct_classified / total_news_test_dataset))


# shuffle_processed_csv('./datasets/dataset_news_classifier_processed.csv')

# classifier = MLClassifier()
# classifier.build_matrix()
# classifier.process_matrix()
# classifier.classify_sentence('Afara este frumos si cald112.!hahah urat esti ma.!>? budget')
#classifier.test_mlclassifier()
