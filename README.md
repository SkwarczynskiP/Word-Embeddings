# Word-Embeddings
 CSI 4180 - Natural Language Processing - Homework 2

This word embeddings project:
- Trains a Continuous Bag of Words (CBOW) and Skip-Gram embedding model using a wikipedia dataset
- Compares the two generated models with two additional pre-trained models (GloVe and Google News) using a variety of queries
- Generates graphs to display negative sentiment distributions for each model based on terms that are usually associated with bias
- Trains and compares the results of a simple logistic regression classifier for a Twitter and Google News dataset

# Files:
 The files in this repository include:
- "hw-2.py" the python script
- "cbow.model" the trained CBOW model
- "skipGram.model" the trained Skip-Gram model

# Files Not Included:
The files in this repository include: The files that are created by running this code, but are not included in this repository due to their size include:
- "cbow.model.syn1neg.npy" file generated when creating the CBOW model
- "cbow.model.mv.vectors.npy" file generated when creating the CBOW model
- "skipGram.model.syn1neg.npy" file generated when creating the Skip-Gram model
- "skipGram.model.mv.vectors.npy" file generated when creating the Skip-Gram model
- "normalizedText.json" file containing the normalized text from the wikipedia dataset

# How to Run:
1. Download the files (Downloading the two trained models is optional)
2. Open the terminal and change directories to the location of the downloaded files
3. Type the command "py hw-2.py" to run

NOTE: The first time you run this code, it will take a long time to load and then train the models