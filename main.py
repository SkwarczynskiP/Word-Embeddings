import os
import re
import gensim
import string
import json
import random
import numpy as np
import scipy.sparse
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import csv
from sklearn import preprocessing
from scipy.special import softmax
from scipy.sparse import csr_matrix
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
import pandas as pd
import plotly.express as px
import plotly.io as py

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)

pairs = [
    ('basketball', 'sport'),
    ('basketball', 'football'),
    ('basketball', 'school'),
    ('basketball', 'laptop'),
    ('basketball', 'dog')
]


def preprocess(text):  # Function used in previous assignments
    # Lowercases each token
    text = text.lower()

    # Additional normalization due to the formatting of the reviews in the dataset
    # Gets rid of the "\n" characters in the dataset
    text = text.replace("\\n", "")

    # Removes all punctuation (referenced https://www.geeksforgeeks.org/python-remove-punctuation-from-string/)
    text = re.sub(r'[^\w\s]', '', text)

    # Removes all stopwords
    stopWords = set(stopwords.words('english'))
    tempList = text.split()
    filteredWords = [tempList for tempList in tempList if tempList.lower() not in stopWords]
    text = (' '.join(filteredWords))

    # Lemmatization of each token (referenced
    # https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/ for different examples)
    lemmatizer = WordNetLemmatizer()
    tempList = text.split()
    lemma = [lemmatizer.lemmatize(tempList) for tempList in tempList]
    finalText = (' '.join(lemma))

    return finalText


def skipGram(modelSkipGram):
    print("Skip-gram Model:")

    # Similarities of the pairs
    print("Word1\tWord2\tSimilarity")
    for word1, word2 in pairs:
        print('%r\t%r\t%.2f' % (word1, word2, modelSkipGram.wv.similarity(word1, word2)))

    print("The top 5 most similar words to the word 'basketball' are:")
    print(modelSkipGram.wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(modelSkipGram.wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(modelSkipGram.wv.most_similar(positive=['basketball', 'college'], negative='professional', topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          modelSkipGram.wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , modelSkipGram.wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def cbow(modelCBOW):
    print("\nCBOW Model:")

    # Similarities of the pairs
    print("Word1\tWord2\tSimilarity")
    for word1, word2 in pairs:
        print('%r\t%r\t%.2f' % (word1, word2, modelCBOW.wv.similarity(word1, word2)))

    print("The top 5 most similar words to the word 'basketball' are:")
    print(modelCBOW.wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(modelCBOW.wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(modelCBOW.wv.most_similar(positive=['basketball', 'college'], negative=['professional'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          modelCBOW.wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , modelCBOW.wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def googleNews(wv):
    print("\nGoogle News Model:")

    # Similarities of the pairs
    print("Word1\tWord2\tSimilarity")
    for word1, word2 in pairs:
        print('%r\t%r\t%.2f' % (word1, word2, wv.similarity(word1, word2)))

    print("The top 5 most similar words to the word 'basketball' are:")
    print(wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(wv.most_similar(positive=['basketball', 'college'], negative=['professional'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def glove(wv):
    print("\nGlove Model:")

    # Similarities of the pairs
    print("Word1\tWord2\tSimilarity")
    for word1, word2 in pairs:
        print('%r\t%r\t%.2f' % (word1, word2, wv.similarity(word1, word2)))

    print("The top 5 most similar words to the word 'basketball' are:")
    print(wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(wv.most_similar(positive=['basketball', 'college'], negative=['professional'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def sgdForMultinomialLRWithCE(X, y, numPasses=5, learningRate=0.1):
    numDataPoints = X.shape[0]
    numClasses = len(set(y))

    numInputs = X.shape[1]
    w = np.zeros((numInputs, numClasses))
    b = np.zeros(numClasses)

    for currentPass in range(numPasses):
        order = list(range(numDataPoints))
        random.shuffle(order)

        for i in order:
            xi = X[i]
            yi = y[i]

            yiOnehot = [0] * numClasses
            yiOnehot[yi] = 1

            z = xi.dot(w) + b
            yHati = softmax(z)

            w = w - learningRate * ((yHati - yiOnehot).T @ xi).T
            b = b - learningRate * (yHati - yiOnehot)

    return w, b


def makePredictionsMultinomial(w, b, X):
    outputs = X.dot(w) + b
    return np.argmax(outputs, axis=1)


def convertToWEFE(model):
    if isinstance(model, gensim.models.Word2Vec):
        wefe = WordEmbeddingModel(model.wv)
    else:
        wefe = WordEmbeddingModel(model)

    return wefe


def sentenceVector(sentence, model):
    vectorSize = model.vector_size
    result = np.zeros(vectorSize)
    count = 1

    for word in sentence:
        if word in model:
            result += model[word]
            count += 1

    result = result / count
    return result


def main():
    # 2.1 Dataset
    normalizedText = []

    if not os.path.exists("normalizedText.json"):
        texts = dataset["train"]["text"]

        # Preprocesses text from the dataset
        for text in texts:
            processedText = preprocess(text)
            normalizedText.append(processedText)

        with open("normalizedText.json", "w") as json_file:
            json.dump(normalizedText, json_file)
    else:
        with open("normalizedText.json", "r") as json_file:
            normalizedText = json.load(json_file)

    # Tokenizing
    normalizedTokens = [sentence.split() for sentence in normalizedText]

    # 2.2 Training Word Embeddings

    # Creating the Skip-gram model
    if not os.path.exists("skipGram.model"):
        modelSkipGram = gensim.models.Word2Vec(normalizedTokens, sg=1, window=5, min_count=1, vector_size=200,
                                               workers=8)
        modelSkipGram.save("skipGram.model")
    else:
        modelSkipGram = gensim.models.Word2Vec.load("skipGram.model")

    # Creating the CBOW model
    if not os.path.exists("cbow.model"):
        modelCBOW = gensim.models.Word2Vec(normalizedTokens, sg=0, window=5, min_count=1, vector_size=200, workers=8)
        modelCBOW.save("cbow.model")
    else:
        modelCBOW = gensim.models.Word2Vec.load("cbow.model")

    # 2.3 Comparing Word Embeddings
    print("Comparing Word Embeddings:\n")
    modelGoogleNews = api.load('word2vec-google-news-300')  # Loading in the Google News model
    modelGlove = api.load('glove-wiki-gigaword-300')  # Loading in the GloVe model

    # Function calls for comparing word embeddings. Each function call runs the 5 queries for the word embeddings
    skipGram(modelSkipGram)
    cbow(modelCBOW)
    googleNews(modelGoogleNews)
    glove(modelGlove)

    # 2.4 Bias in Word Embeddings
    # RNSBWords = [["swedish"], ["irish"], ["mexican"], ["chinese"], ["filipino"], ["german"], ["english"], ["french"],
    #              ["norwegian"], ["american"], ["indian"], ["dutch"], ["russian"], ["scottish"], ["italian"]]
    # biasWords = [["homemaker"], ["boss"], ["nurse"], ["doctor"], ["teacher"], ["engineer"], ["dancer"],
    #              ["construction"], ["assistant"], ["programmer"], ["saleswoman"], ["athlete"]]
    #
    # bing_liu = load_bingliu()
    # query = Query(RNSBWords, [bing_liu["positive_words"], bing_liu["negative_words"]])
    # biasQuery = Query(biasWords, [bing_liu["positive_words"], bing_liu["negative_words"]])
    #
    # myModels = [(modelCBOW, "cbow"), (modelSkipGram, "skipGram"), (modelGoogleNews, "googleNews"), (modelGlove, "glove")]
    # myQueries = [query, biasQuery]
    #
    # for model, modelName in myModels:
    #     modelWefe = convertToWEFE(model)
    #
    #     for i, query in enumerate(myQueries, 1):
    #         result = RNSB().run_query(query, modelWefe, lost_vocabulary_threshold=0.28)
    #
    #         df_negative = pd.DataFrame(list(result['negative_sentiment_distribution'].items()),
    #                                    columns=['word', 'negative_sentiment_distribution'])
    #
    #         # Plot the results
    #         fig = px.bar(df_negative, x='word', y='negative_sentiment_distribution',
    #                      title=f"Negative Sentiment Distribution for {modelName}",
    #                      labels={"negative_sentiment_distribution": "Negative Sentiment Distribution", "word": "Word"})
    #         fig.update_yaxes(range=[0, 0.2])
    #         fig.show()

    # 2.5 Text Classification
    print("\nText Classification:")
    texts = []
    labels = []

    # Loading the dataset
    tweets = load_dataset('osanseviero/twitter-airline-sentiment')

    for tweet in tweets['train']:
        texts.append(tweet['text'])
        labels.append(tweet['airline_sentiment'])

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)

    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    X = vectorizer.fit_transform(texts)

    mw, mb = sgdForMultinomialLRWithCE(X, y)

    preds = makePredictionsMultinomial(mw, mb, X)
    print(classification_report(y, preds))

    # 2ND LINEAR REGRESSION MODEL
    sentencesVectors = []
    for sentence in texts:
        sentence = preprocess(sentence)
        sentencesVectors.append(sentenceVector(sentence, modelGoogleNews))

    X = np.array(sentencesVectors)
    X = scipy.sparse.csr_matrix(X)

    mw, mb = sgdForMultinomialLRWithCE(X, y)

    preds = makePredictionsMultinomial(mw, mb, X)
    print(classification_report(y, preds))


if __name__ == "__main__":
    main()
