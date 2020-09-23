Implementation of two baseline methods(TF-IDF and BOW), and a proposed method word2vec, doc2vec
This implementation requires Python >=3.0, detailed comments can be find in p1.py.

function:
1) evaluation: get the evaluation metrics
2) baseline1, baseline2, baseline3, proposed: use the vector obtained for nlp methods to train and test a model
3) word2vec, tf_idf, BOW: build vectors with different nlp methods
4) DataLoader: load data from xlsx files and call the preprocessor
5) preprocessor: tokenize, remove punctuations or stopwords and stem the review docs
6) dict: build the Dictionary with tokenized sentences
7) main: main function
8) writter: write the output dataframe into a csv file as required in the homework instrctions

To run a method, set method = 1 when calling the main function, other wise let method = 0