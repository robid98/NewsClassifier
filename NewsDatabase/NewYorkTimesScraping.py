# In this clss I will scrap the New York Times API and search for articles from 2017. After that I will
# add them in a MongoDb instance for further processing.
import time
import requests
from NewsDatabase.ProcessNews import ProccessNews, lemmatize_words
from NewsDatabase.StoreMongoDb import StoreMongoDb


def check_valid_respponse(article, year, month):
    headline = article['headline']['main']
    snippet = article['snippet']
    document_type = article['document_type']
    news_desk = article['news_desk']

    ok_article = 0
    if 'headline' in article and 'snippet' in article and 'document_type' in \
            article and 'news_desk' in article and 'keywords' in article:
        if len(headline) and len(snippet) and document_type == 'article' and len(news_desk):
            ok_article = 1
        else:
            ok_article = 0

    if ok_article:
        date_splitted = article['pub_date'].split('-')
        y = int(date_splitted[0])
        m = int(date_splitted[1])

        if year == y and month == m:
            ok_article = 1
        else:
            ok_article = 0

    return ok_article


class NewYorkTimesScraping:
    def __init__(self):
        self.api_key = ""
        self.api_get_url = "https://api.nytimes.com/svc/archive/v1/"
        self.list_dates = []
        self.process_news = ProccessNews()
        self.mongo_db = StoreMongoDb()

    def get_dates(self):
        years = [2015, 2016, 2017]
        for i in range(0, 3):
            for j in range(1, 13):
                self.list_dates.append((j, years[i]))

    def get_dates_for_test_classifier(self):
        years = [2018]
        for i in range(0, 1):
            for j in range(1, 13):
                self.list_dates.append((j, years[i]))

    def api_request(self, year, month):
        """Send GET request to New York Times archive API and return the response"""
        api_response = requests.get(
            self.api_get_url + "/{}/{}.json?api-key={}".format(year, month, self.api_key)).json()
        time.sleep(6)

        return api_response

    def get_all_articles(self):

        """In this function we will iterate over all dates 2017, 2016, 2015 and check if a article is valid
        If a article is valid we will store them in a MongoDb instance"""
        # self.get_dates()  # training classifier 2015, 2016, 2017
        self.get_dates_for_test_classifier()  # test 2018

        article_number = 0
        for month, year in self.list_dates:
            all_month_documents = self.api_request(year, month)
            for document in all_month_documents['response']['docs']:
                if check_valid_respponse(document, year, month):
                    article_prototype = {'content': "", 'document_type': "", 'category': "", 'keywords': [], 'date': ""}
                    headline = document['headline']['main']
                    snippet = document['snippet']
                    # That means we have a valid article, we should process the text
                    # Lemmatize, remove stopwords, punctuation ..
                    article_content = lemmatize_words(self.process_news.remove_stop_words
                        ((self.process_news.remove_punctuation(
                        headline + " " + snippet))))
                    article_prototype["content"] = article_content
                    article_prototype["document_type"] = document["document_type"]
                    article_prototype["category"] = document["news_desk"]
                    article_prototype["date"] = document["pub_date"]

                    for keyword in document["keywords"]:
                        article_prototype["keywords"].append((keyword['name'], keyword['value']))

                    # self.mongo_db.insert_news_new_york_times_collection(article_prototype)
                    self.mongo_db.insert_news_new_york_times_test(article_prototype)

                    article_number += 1
                    print(article_number)


obj = NewYorkTimesScraping()
# obj.get_all_articles()
# obj.get_dates_for_test_classifier()
# print(obj.list_dates)
