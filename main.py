import re
import os
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)


def preprocess(text):
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


def main():
    # check the first example of the training portion of the dataset:
    # print(dataset["train"][0])

    normalizedText = []

    if not os.path.exists("normalizedText.json"):
        texts = dataset["train"]["text"]

        # Preprocesses text from the dataset
        for text in texts:
            processedText = preprocess(text)
            normalizedText.append(processedText)

        with open("normalizedText", "w") as json_file:
            json.dump(normalizedText, json_file)
    else:
        with open("normalizedText", "w") as json_file:
            normalizedText = json.load(json_file)

    # print(normalizedText[0]) # Test to make sure it is storing data correctly (GOOD)

    if not os.path.exists("skip_gram.model"):
        modelSkipGram = Word2Vec(normalizedText, sg=1, window=5, min_count=5, workers=4)
        modelSkipGram.wv.save_word2vec_format('skip_gram.model')
    else:
        modelSkipGram = Word2Vec.load('skip_gram.model')

    if not os.path.exists('cbow.model'):
        modelCBOW = Word2Vec(normalizedText, sg=0, window=5, min_count=5, workers=4)
        modelCBOW.wv.save_word2vec_format('cbow.model')
    else:
        modelCBOW = Word2Vec.load('cbow.model')


if __name__ == "__main__":
    main()
