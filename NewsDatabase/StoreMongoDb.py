# This class is to make the news received from the datasets persisent with a MongoDb database
import csv
import os

import pymongo
from NewsDatabase.GetNews import read_news_from_dataset

BASE = os.path.dirname(os.path.abspath(__file__))


class StoreMongoDb:
    def __init__(self):
        self.newsClient = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.newsClient["newsDatabase"]
        self.collection = self.db["news"]
        self.collection_scraped = self.db["new_york_times_scraped"]
        self.collection_scraped_test = self.db["new_york_times_scraped_test"]

    def insert_news(self):
        true_news = read_news_from_dataset('datasets/True_News.csv')
        false_news = read_news_from_dataset('datasets/Fake_News.csv')
        for index, news in enumerate(true_news):
            self.collection.insert_one(news)
        for index, news in enumerate(false_news):
            self.collection.insert_one(news)

    def insert_news_new_york_times_collection(self, article_prototype):
        self.collection_scraped.insert_one(article_prototype)

    def insert_news_new_york_times_test(self, article_prototype):
        self.collection_scraped_test.insert_one(article_prototype)

    def read_data_and_create_csv(self):

        news_database = self.collection.find()
        vocabulary_set = set()

        # read vocabulary and create csv only with relevant words
        vocabulary = open(os.path.join(BASE, "datasets\\vocabulary.txt"), 'r')
        for line in vocabulary:
            vocabulary_set.add(line.rstrip("\n"))

        with open(os.path.join(BASE, "datasets\\dataset_news_classifier_processed.csv"), 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "news_text", "classification"])

            counter = 0
            classified = 0
            for news in news_database:
                counter = counter + 1

                if news['classified'] == 'True':
                    classified = 1
                else:
                    classified = 0

                tokenize_resulted_text = news['resultedText'].split()
                resulted_list_after_selection = []
                for word in tokenize_resulted_text:
                    if word in vocabulary_set:
                        resulted_list_after_selection.append(word)

                csv_row = [counter, ' '.join(resulted_list_after_selection), classified]
                writer.writerow(csv_row)

    def read_data_and_create_csv_scraped(self):
        # scraped_news = self.collection_scraped.find()
        scraped_news = self.collection_scraped_test.find()

        with open(os.path.join(BASE, "datasets_scraped\\dataset_news_classifier_scraped_test.csv"), 'w', newline='',
                  encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "news_content", "category"])

            counter = 0
            for news in scraped_news:
                counter = counter + 1

                csv_row = [counter, news['content'], news['category']]
                writer.writerow(csv_row)

# db = StoreMongoDb()
# db.insert_news()
# db.read_data_and_create_csv()
# db.read_data_and_create_csv_scraped()
