# This class is used for getting the relevant news and add them into a MongoDb database
# For this we use the two datasets for true and fake news from
# https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
"""The news are obtained from different legitimate news sites and sites flagged as unreliable by Politifact.com"""
import csv
from NewsDatabase.ProcessNews import ProccessNews, lemmatize_words


def read_news_from_dataset(dataset_file_path):
    news_list = []
    news_format = {"subject": None, "resultedText": None, "date": None, "classified": None}
    proccessNews = ProccessNews()  # helper functions for text proccesing
    csv_line = 0

    with open(dataset_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            csv_line += 1
            if csv_line == 1:
                pass
            else:
                news_text = lemmatize_words(proccessNews.remove_stop_words((proccessNews.remove_punctuation(row[1]))))
                news_title = lemmatize_words(proccessNews.remove_stop_words((proccessNews.remove_punctuation(row[0]))))
                news_format['subject'] = row[2]
                news_format['resultedText'] = news_title + ' ' + news_text
                news_format['date'] = row[3]
                if 'True' in dataset_file_path:
                    news_format['classified'] = 'True'
                else:
                    news_format['classified'] = 'False'
                news_list.append(news_format.copy())
    return news_list
