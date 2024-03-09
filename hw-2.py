import os
import re
import gensim
import json
import random
import scipy
import numpy as np
import pandas as pd
import plotly.express as px
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.special import softmax
from scipy.sparse import csr_matrix
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)


def preprocess(text):  # Function used in previous assignments
    text = text.lower()  # Lowercases each token
    text = text.replace("\\n", "")  # Gets rid of the "\n" characters in the dataset

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

    print("The top 5 most similar words to the word 'basketball' are: ")
    print(modelSkipGram.wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(modelSkipGram.wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(modelSkipGram.wv.most_similar(positive=['basketball', 'college'], negative='professional', topn=5))

    # Top 5 similar words to a pair of words that are not that similar
    print("The top 5 most similar words to the pair of words 'basketball' and 'laptop' are:")
    print(modelSkipGram.wv.most_similar(positive=['basketball', 'laptop'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          modelSkipGram.wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , modelSkipGram.wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def cbow(modelCBOW):
    print("\nCBOW Model:")

    print("The top 5 most similar words to the word 'basketball' are:")
    print(modelCBOW.wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(modelCBOW.wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(modelCBOW.wv.most_similar(positive=['basketball', 'college'], negative=['professional'], topn=5))

    # Top 5 similar words to a pair of words that are not that similar
    print("The top 5 most similar words to the pair of words 'basketball' and 'laptop' are:")
    print(modelCBOW.wv.most_similar(positive=['basketball', 'laptop'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          modelCBOW.wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , modelCBOW.wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def googleNews(wv):
    print("\nGoogle News Model:")

    print("The top 5 most similar words to the word 'basketball' are:")
    print(wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(wv.most_similar(positive=['basketball', 'college'], negative=['professional'], topn=5))

    # Top 5 similar words to a pair of words that are not that similar
    print("The top 5 most similar words to the pair of words 'basketball' and 'laptop' are:")
    print(wv.most_similar(positive=['basketball', 'laptop'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def glove(wv):
    print("\nGlove Model:")

    print("The top 5 most similar words to the word 'basketball' are:")
    print(wv.most_similar(positive='basketball', topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college' are:")
    print(wv.most_similar(positive=['basketball', 'college'], topn=5))

    print("The top 5 most similar words to the pair of words 'basketball' and 'college', but minus the word "
          "'professional' are:")
    print(wv.most_similar(positive=['basketball', 'college'], negative=['professional'], topn=5))

    # Top 5 similar words to a pair of words that are not that similar
    print("The top 5 most similar words to the pair of words 'basketball' and 'laptop' are:")
    print(wv.most_similar(positive=['basketball', 'laptop'], topn=5))

    # Picks the word that does not match, where there is only one obvious choice
    print("The word that does not match in the list ['basketball', 'sports', 'game', 'college', 'water', 'cat'] is: ",
          wv.doesnt_match(['basketball', 'sports', 'game', 'college', 'water', 'cat']))

    # Picks the word that does not match, where there are two choices that could be picked
    print("The word that does not match in the list ['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat'] is: "
          , wv.doesnt_match(['basketball', 'soccer', 'baseball', 'football', 'dog', 'cat']))


def sgdForMultinomialLRWithCE(X, y, numPasses=5, learningRate=0.1):
    # Entire function referenced from the class Google Collab Notebook, Lecture 12
    numDataPoints = X.shape[0]
    numClasses = len(set(y))

    numInputs = X.shape[1]
    w = np.zeros((numInputs, numClasses))
    b = np.zeros(numClasses)

    for currentPass in range(numPasses):

        # Randomly iterates through the data points
        order = list(range(numDataPoints))
        random.shuffle(order)

        for i in order:
            # Computes y-hat for the value of i, given yi and xi
            xi = X[i]
            yi = y[i]

            yiOnehot = [0] * numClasses
            yiOnehot[yi] = 1

            z = xi.dot(w) + b
            yHati = softmax(z)

            # Updates the weights and biases for each w and b
            w = w - learningRate * ((yHati - yiOnehot).T @ xi).T
            b = b - learningRate * (yHati - yiOnehot)

    return w, b


def makePredictionsMultinomial(w, b, X):
    # Entire function referenced from the class Google Collab Notebook, Lecture 12
    outputs = X.dot(w) + b
    return np.argmax(outputs, axis=1)


def convertToWEFE(model):
    # Function to convert the model to a WEFE model (referenced code generated by Microsoft Copilot)
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

    # 2.2 Training Word Embeddings (referenced
    #   https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists-2/ for how to check if a file exists)
    if not os.path.exists("skipGram.model"):  # Creating the Skip-Gram model
        modelSkipGram = gensim.models.Word2Vec(normalizedTokens, sg=1, window=5, min_count=1,
                                               vector_size=200, workers=8)
        modelSkipGram.save("skipGram.model")
    else:  # Loading the Skip-Gram model
        modelSkipGram = gensim.models.Word2Vec.load("skipGram.model")

    if not os.path.exists("cbow.model"):  # Creating the CBOW model
        modelCBOW = gensim.models.Word2Vec(normalizedTokens, sg=0, window=5, min_count=1, vector_size=200, workers=8)
        modelCBOW.save("cbow.model")
    else:  # Loading the CBOW model
        modelCBOW = gensim.models.Word2Vec.load("cbow.model")

    # 2.3 Comparing Word Embeddings
    print("Comparing Word Embeddings:\n")
    modelGoogleNews = api.load('word2vec-google-news-300')  # Loading the Google News model
    modelGlove = api.load('glove-wiki-gigaword-300')  # Loading the GloVe model

    # Function calls for comparing word embeddings. Each function call runs the 5 queries for the word embeddings
    skipGram(modelSkipGram)
    cbow(modelCBOW)
    googleNews(modelGoogleNews)
    glove(modelGlove)

    # 2.4 Bias in Word Embeddings (referenced https://www.w3schools.com/python/pandas/default.asp for how to use pandas,
    #   and https://wefe.readthedocs.io/en/latest/examples/replications.html
    #   and https://plotly.com/python/bar-charts/ for how to create bar charts)
    RNSBWords = [["swedish"], ["irish"], ["mexican"], ["chinese"], ["filipino"], ["german"], ["english"], ["french"],
                 ["norwegian"], ["american"], ["indian"], ["dutch"], ["russian"], ["scottish"], ["italian"]]
    biasWords = [["homemaker"], ["boss"], ["nurse"], ["doctor"], ["teacher"], ["engineer"], ["dancer"],
                 ["construction"], ["assistant"], ["programmer"], ["saleswoman"], ["athlete"]]

    bing_liu = load_bingliu()

    # Query for RNSB Words. RNSB Words highlight different nationalities
    query = Query(RNSBWords, [bing_liu["positive_words"], bing_liu["negative_words"]])

    # Query for my biased words. My biased words highlight professions that are commonly attributed to a specific gender
    biasQuery = Query(biasWords, [bing_liu["positive_words"], bing_liu["negative_words"]])

    myModels = [(modelCBOW, "Continuous Bag of Words"), (modelSkipGram, "Skip-Gram"), (modelGoogleNews, "Google News"),
                (modelGlove, "GloVe")]
    myQueries = [query, biasQuery]

    # Loops through each of the models and queries to generate the graphs
    for model, modelName in myModels:
        modelWefe = convertToWEFE(model)  # Converts the specific model to a WEFE model

        for i, query in enumerate(myQueries, 1):
            result = RNSB().run_query(query, modelWefe, lost_vocabulary_threshold=0.28)

            df_negative = pd.DataFrame(list(result['negative_sentiment_distribution'].items()),
                                       columns=['word', 'negative_sentiment_distribution'])

            # Creates the result graph for each of the different models and queries
            fig = px.bar(df_negative, x='word', y='negative_sentiment_distribution',
                         title=f"Negative Sentiment Distribution: {modelName}",
                         labels={"negative_sentiment_distribution": "Negative Sentiment Distribution", "word": "Word"})
            fig.update_yaxes(range=[0, 0.2])
            fig.show()

    # 2.5 Text Classification (code referenced from the class Google Collab Notebook, Lecture 12)
    print("\nText Classification:\n")
    texts = []
    labels = []

    tweets = load_dataset('osanseviero/twitter-airline-sentiment')  # Loading the dataset

    # 1st Linear Regression Model (on twitter airline sentiment dataset)
    for tweet in tweets['train']:
        texts.append(tweet['text'])
        labels.append(tweet['airline_sentiment'])

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)

    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    X = vectorizer.fit_transform(texts)

    mw, mb = sgdForMultinomialLRWithCE(X, y)

    preds = makePredictionsMultinomial(mw, mb, X)
    print("Twitter Airline Sentiment Dataset (Linear Regression Model):")
    print(classification_report(y, preds))

    # 2nd Linear Regression Model (on Google News dataset)
    sentencesVectors = []
    for sentence in texts:
        sentence = preprocess(sentence)
        sentencesVectors.append(sentenceVector(sentence, modelGoogleNews))

    X = np.array(sentencesVectors)
    X = scipy.sparse.csr_matrix(X)

    mw, mb = sgdForMultinomialLRWithCE(X, y)

    preds = makePredictionsMultinomial(mw, mb, X)
    print("Google News Dataset (Linear Regression Model):")
    print(classification_report(y, preds))


if __name__ == "__main__":
    main()
