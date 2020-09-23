import numpy as np
import gensim.models.word2vec as word2vec
from gensim.parsing.preprocessing import remove_stopwords
from gensim.test.utils import common_texts
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from gensim import corpora, matutils
from gensim.models import TfidfModel, Word2Vec, doc2vec
import nltk
from nltk.tokenize import word_tokenize
import string
import pdb
import pandas as pd


class Solution(object):

    def evaluation(self, predictions):
        '''
        :param predictions: predictED labels
        :return: show the performance metrics
        '''
        print(classification_report(self.df_testing['label'], predictions))
        accuracy = accuracy_score(self.df_testing['label'], predictions)
        print('accuracy is:', accuracy)
        f1_macro = f1_score(self.df_testing['label'], predictions, average='macro')
        f1_micro = f1_score(self.df_testing['label'], predictions, average='micro')
        f1_weighted = f1_score(self.df_testing['label'], predictions, average='weighted')
        print('macro f1 is:', f1_macro)
        print('micro f1 is:', f1_micro)
        print('weighted f1 is:', f1_weighted)

    def baseline1(self):
        '''
        :return: results of tfidf
        '''
        tfidf_training = self.tf_idf(self.df_training)
        tfidf_testing = self.tf_idf(self.df_testing)
        # initialize the classifer
        clf_decision_tfidf = DecisionTreeClassifier(random_state=2)
        clf_decision_tfidf.fit(tfidf_training, self.training_y)
        predictions_tfidf = clf_decision_tfidf.predict(tfidf_testing)
        print('*'*9, 'performance of TF-IDF', '*'*9)
        self.evaluation(predictions_tfidf)
        self.writter(predictions_tfidf, file_name='testing_output_tfidf.csv')

    def baseline2(self):
        '''
        :return: results of BOW
        '''
        BOW_training = self.BOW(self.df_training)
        BOW_testing = self.BOW(self.df_testing)
        # initialize the classifer
        clf_decision_BOW = DecisionTreeClassifier(random_state=2)
        clf_decision_BOW.fit(BOW_training, self.training_y)
        predictions_BOW = clf_decision_BOW.predict(BOW_testing)
        print('*' * 9, 'performance of Bag of Word', '*' * 9)
        self.evaluation(predictions_BOW)
        self.writter(predictions_BOW, file_name='testing_output_bow.csv')

    def proposed(self):
        '''
        :return: results of CBOW
        '''
        w2v_training = self.word2vec(self.df_training)
        w2v_testing = self.word2vec(self.df_testing)
        # initialize the classifer
        clf_decision_w2v = DecisionTreeClassifier(random_state=2)
        clf_decision_w2v.fit(w2v_training, self.training_y)
        predictions_w2v = clf_decision_w2v.predict(w2v_testing)
        print('*' * 9, 'performance of Word2Vec(CBOW)', '*' * 9)
        self.evaluation(predictions_w2v)
        self.writter(predictions_w2v, file_name='testing_output_word2vec.csv')

    def word2vec(self, data):
        '''
        :param data: training or testing dataframe
        :return: word2vec vectors
        '''
        self.size = 1000      # number of dimensions of the embeddings
        self.window = 3       # maximum window distance around a target word
        self.min_count = 1    # threshold occurrence to ignore a word
        self.workers = 3      # number of partitions during training
        self.sg = 0           # 0: CBOW; 1: skip gram
        stemmed_tokens = self.content.values
        # train the Word2Vec model
        w2v_model = Word2Vec(stemmed_tokens, min_count=self.min_count, size=self.size, workers=self.workers,
                             window=self.window, sg=self.sg)
        # generate the CBOW vector
        w2v_metrics = []
        for index, line in data.iterrows():
            features = (np.mean([w2v_model[token] for token in line['tokenized_text']], axis=0)).tolist()
            w2v_metrics.append(features)
        return pd.DataFrame(w2v_metrics)

    def tf_idf(self, data):
        '''
        :param data: training or testing dataframe
        :return: tfidf vectors
        '''
        # build the TF-IDF model
        corpus = [self.mydict.doc2bow(line) for line in self.content]
        self.tfidf_model = TfidfModel(corpus)
        # generate the TF-IDF vector
        tfidf_metrics = []
        for index, line in data.iterrows():
            doc = self.mydict.doc2bow(line['tokenized_text'])
            features = matutils.corpus2csc([self.tfidf_model[doc]], num_terms=self.vocab_len).toarray()[:, 0]
            tfidf_metrics.append(features)

        return pd.DataFrame(tfidf_metrics)

    def BOW(self, data):
        '''
        :param data: training or testing dataframe
        :return: BOW vectors
        '''
        # corpus = [self.mydict.doc2bow(line) for line in self.content]
        BOW_metrics = []
        # generate the BOW vector
        for index, line in data.iterrows():
            doc = self.mydict.doc2bow(line['tokenized_text'])
            features = matutils.corpus2csc([doc], num_terms=self.vocab_len).toarray()[:, 0]
            BOW_metrics.append(features)
        return pd.DataFrame(BOW_metrics)

    def DataLoader(self, path):
        '''
        :param path: path to training and testing files
        :return: preprocessed text and corresponding labels
        '''
        print(">> loading data")
        self.train_file = path + 'P1_training.xlsx'
        self.test_file = path + 'P1_testing.xlsx'
        self.df_training = pd.read_excel(self.train_file, sheet_name='Sheet1')
        self.df_testing = pd.read_excel(self.test_file, sheet_name='P1_testing')
        self.training_x, self.testing_x = self.df_training['sentence'], self.df_testing['sentence']
        self.training_y, self.testing_y = self.df_training['label'], self.df_testing['label']
        print('>> length of training data', len(self.df_training))
        print('distribution of training dataset:\n', self.training_y.value_counts())
        print('>> length of testing data', len(self.df_testing))
        print('distribution of testing dataset:\n', self.testing_y.value_counts())

        self.df_training['tokenized_text'] = [self.preprocessor(text) for text in self.training_x]
        self.df_testing['tokenized_text'] = [self.preprocessor(text) for text in self.testing_x]

    def preprocessor(self, text):
        '''
        :param text: original review text
        :return: preprocessed text after tokenization, removing stop words and punctuations and stemming
        '''
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [w.lower() for w in tokens]    # convert to lower case
        # removing unnecessary punctuations or stopwords
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [w for w in stripped if w.isalpha()]

        # words = [w for w in tokens if w.isalpha]
        stop_words = nltk.corpus.stopwords.words('english')
        words = [w for w in words if w not in stop_words]

        porter = nltk.stem.PorterStemmer()
        words_stemmed = [porter.stem(w) for w in words]            # stemming

        return words_stemmed

    def dict(self):
        '''
        :return: build Dictionary with tokenized setences
        '''
        self.content = pd.concat([self.df_training['tokenized_text'], self.df_testing['tokenized_text']])
        self.mydict = corpora.Dictionary(self.content)
        self.vocab_len = len(self.mydict.token2id)
        # self.mydict = corpora.Dictionary(self.df_training['tokenized_text'])
        print('number of total unique words:', self.vocab_len)

    def writter(self, predictions, file_name):
        self.df_testing['gold_label'] = self.df_testing['label']
        self.df_testing['predicted_label'] = pd.Series(predictions)
        df_output = self.df_testing[['sentence', 'gold_label', 'predicted_label']]
        df_output.to_csv('./P1/'+file_name, index=False)

        return


def main(baseline1=1, baseline2=1, proposed=1):
    s = Solution()
    s.DataLoader('./P1/')
    s.dict()
    if baseline1 == True:
        s.baseline1()
    if baseline2 == True:
        s.baseline2()
    if proposed == True:
        s.proposed()


if __name__ == '__main__':
    """
    baseline1: tfidf
    baseline2: BOW
    proposed: CBOW
    0 ---> don't run this method; 1 ---> run this method
    """
    main(baseline1=1, baseline2=1, proposed=1)
