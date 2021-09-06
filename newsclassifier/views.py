import os

from django.shortcuts import render
from NewsDatabase.MLClassifierScraped import MLClassifierScraped
from NewsDatabase.KNNAlgorithm import KNNAlgorithm
BASE = os.path.dirname(os.path.abspath(__file__))


def home(request):
    if request.method == 'POST':
        article_content = request.POST.get('article_content')
        # MlClassifier Object
        mlcassifier_object = MLClassifierScraped()
        knn_object = KNNAlgorithm()
        # Classify the article content received from the Form
        category_probabilities, category_classified = mlcassifier_object.classify_sentence(article_content, 1)
        category_probabilities_bag_of_words, category_classified_bag_of_words = mlcassifier_object.classify_sentence(
            article_content, 2)
        category_knn = knn_object.knn_algorithm(article_content, 5)

        category_probabilities = {k: str(v) for k, v in
                                  sorted(category_probabilities.items(), key=lambda item: item[1], reverse=True)}

        category_probabilities_bag_of_words = {k: str(v) for k, v in
                                               sorted(category_probabilities_bag_of_words.items(), key=lambda item: item[1],
                                                      reverse=True)}

        first_probabilities = {}
        for index, category in enumerate(category_probabilities):
            first_probabilities[category] = category_probabilities[category]
            if index == 2:
                break

        first_probabilities_bag_of_word = {}
        for index, category in enumerate(category_probabilities_bag_of_words):
            first_probabilities_bag_of_word[category] = category_probabilities_bag_of_words[category]
            if index == 2:
                break

        context = {
            'article_content': article_content,
            'category_classified': category_classified,
            'category_probabilities': first_probabilities,
            'category_classified_bag_of_words': category_classified_bag_of_words,
            'category_probabailities_bag_of_words':  first_probabilities_bag_of_word,
            'category_knn': category_knn
        }
        return render(request, 'django_news_classifier/classifierlogic.html', context)

    return render(request, 'django_news_classifier/classifierlogic.html')
