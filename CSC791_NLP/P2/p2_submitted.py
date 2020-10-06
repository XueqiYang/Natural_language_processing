import gensim
import nltk
import string
import pandas as pd
import numpy as np
import math
import re
from sklearn import svm
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
from sklearn import preprocessing
from collections import namedtuple
from sentence_transformers import SentenceTransformer
from keras.preprocessing.sequence import pad_sequences


class Solution(object):

    def evaluation(self, predictions):
        '''
        :param predictions: predictED labels
        :return: show the performance metrics
        '''
        print(classification_report(self.testing_y, predictions))
        accuracy = accuracy_score(self.testing_y, predictions)
        print('accuracy is:', accuracy)
        f1_macro = f1_score(self.testing_y, predictions, average='macro')
        f1_micro = f1_score(self.testing_y, predictions, average='micro')
        f1_weighted = f1_score(self.testing_y, predictions, average='weighted')
        print('macro f1 is:', f1_macro)
        print('micro f1 is:', f1_micro)
        print('weighted f1 is:', f1_weighted)

    def DataLoader(self):
        '''
        :param path: path to training and testing files
        :return: preprocessed text and corresponding labels
        '''
        print(">> loading data")
        self.train_file = 'p2_train.csv'
        self.test_file = 'p2_test.csv'
        self.df_training = pd.read_csv(self.train_file)       # 1640
        self.df_testing = pd.read_csv(self.test_file)         # 410
        self.split = len(self.df_training)
        self.df = pd.concat([self.df_training, self.df_testing])     # join two df to process together
        self.encoder_label()
        self.concatenate()
        self.df['tokenized'] = [self.preprocessor(text) for text in self.df['content']]
        self.training_x, self.testing_x = self.df['tokenized'].iloc[:self.split], self.df['tokenized'].iloc[self.split:]
        #

        self.training_y, self.testing_y = self.df['label'].iloc[:self.split], self.df['label'].iloc[self.split:]
        print('>> length of training data', self.split)
        print('distribution of training dataset:\n', self.training_y.value_counts())
        print('>> length of testing data', len(self.df_testing))
        print('distribution of testing dataset:\n', self.testing_y.value_counts())

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

    def encoder_label(self):
        '''
        :return: encoded labels: 'agreed': 0 , 'answered': 1, 'attacked': 2, 'irrelevant': 3
        '''
        self.type = list(self.df['type'])
        stats = Counter(self.type).keys()
        distribution = Counter(self.type).values()
        print('types', stats)
        print('distribution', distribution)
        # #########

        le = preprocessing.LabelEncoder()
        le.fit(self.df['type'])
        print(list(le.classes_))

        list_label = le.transform(self.type)
        self.df['label'] = pd.Series(list_label)
        return

    def concatenate(self):
        series_list = [self.df['question'].astype(str), self.df['subsequent'].astype(str), self.df['response'].astype(str)]
        for series in series_list:
            self.df['content'] = self.df['precedent'].astype(str).str.cat(series, sep=' ')
        return

    def doc2vec(self):
        """
        sentence embedding
        :return:
        """
        self.sentence_embedding()
        self.model_d2v_training = Doc2Vec(self.docs_d2v[:self.split], vector_size=100, window=300, min_count=1, workers=4)
        self.model_d2v_testing = Doc2Vec(self.docs_d2v[self.split:], vector_size=100, window=300, min_count=1, workers=4)
        # #######  make note
        # vector_size: Dimensionality of the feature vectors.
        # self.model_d2v.wv.vocab       # Print model vocabulary
        # print(self.model_d2v_training.docvecs[0])    # numpy.ndarray
        self.vecs_d2v_training = self.model_d2v_training.docvecs.vectors_docs
        self.vecs_d2v_testing = self.model_d2v_testing.docvecs.vectors_docs
        self.vec_d2v = self.normalize(np.concatenate((self.vecs_d2v_training, self.vecs_d2v_testing), axis=0))
        return

    def sentence_embedding(self):
        '''
        sentence_embedding: doc2vec
        :return:
        '''
        self.docs_d2v = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        s = self.df['tokenized']
        for index, text in s.items():
            words = text
            tags = [index]
            self.docs_d2v.append(analyzedDocument(words, tags))
        return

    def pos_tagging(self):
        '''
        :return: a list of tagged tokens in POS tagging
        '''
        wordlist = self.df['tokenized']
        taggeds = []
        for item in wordlist:
            taggeds.append(nltk.pos_tag(item))

        # seperate the tokens and tags
        sentences, sentence_tags = [], []
        for tagged in taggeds:
            sentence, tags = zip(*tagged)
            sentences.append(np.array(sentence))
            sentence_tags.append(np.array(tags))

        return sentences, sentence_tags

    def encoder_pos(self, sentences, sentence_tags):
        '''
        :return: convert to numerical by indexing them in a dictionary
                 assign to each word (and tag) a unique integer
                padding the sequences to get unique lenghth of vectors
        '''
        # OOV – Out Of Vocabulary
        words, tags = set([]), set([])
        for s in sentences:
            for w in s:
                words.add(w.lower())

        for ts in sentence_tags:
            for t in ts:
                tags.add(t)

        word2index = {w: i + 2 for i, w in enumerate(list(words))}
        word2index['-PAD-'] = 0  # The special value used for padding
        word2index['-OOV-'] = 1  # The special value used for OOVs

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2index['-PAD-'] = 0  # The special value used to padding

        # convert word dataset to integer dataset
        sentences_X, tags_y = [], []
        for s in sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])
            sentences_X.append(s_int)

        for s in sentence_tags:
            tags_y.append([tag2index[t] for t in s])
        # convert to fixed sized vectors
        len_max = len(max(sentences_X, key=len))
        self.sentences_X = pad_sequences(sentences_X, maxlen=len_max, padding='post')
        self.tags_y = pad_sequences(tags_y, maxlen=len_max, padding='post')
        self.vec_s_pos = np.array(self.normalize(self.sentences_X))
        self.vec_t_pos = np.array(self.normalize(self.tags_y))      # 2050*684
        return

    def normalize(self, vec):
        '''
        :return: normalized vectors
        '''
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        else:
            return vec/norm

    def proposed_f1(self):
        questions_list = [self.df['question'].astype(str), self.df['subsequent'].astype(str)]
        for series in questions_list:
            df_q = self.df['precedent'].astype(str).str.cat(series, sep=' ')
        df_r = self.df['response'].astype(str)

        for index in range(len(df_q)):
            try:
                df_q.iloc[index] = df_q.iloc[index].to_string()
            except AttributeError:
                df_q.iloc[index] = df_q.iloc[index]

        for index in range(len(df_r)):
            try:
                df_r.iloc[index] = df_r.iloc[index].to_string()
            except AttributeError:
                df_r.iloc[index] = df_r.iloc[index]

        dis_cos = []
        for index in range(len(df_r)):
            cosines = []
            vec1 = self.text2vec(df_q.iloc[index])
            vec2 = self.text2vec(df_r.iloc[index])
            cosine = self.cosine_dis(vec1, vec2)
            cosines.append(cosine)
            dis_cos.append(cosines)
        self.dis_cos = np.array(dis_cos)
        return

    def cosine_dis(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text2vec(self, text):
        WORD = re.compile(r"\w+")
        words = WORD.findall(text)
        return Counter(words)

    def baseline(self):
        '''
        :return: results based on two baseline feature extraction methods
        '''
        self.doc2vec()
        sentences, sentence_tags = self.pos_tagging()
        self.encoder_pos(sentences, sentence_tags)
        features_pos = np.concatenate((self.vec_t_pos, self.vec_s_pos), axis=1)
        self.features1 = np.concatenate((self.vec_d2v, features_pos), axis=1)
        # self.vec_d2v + self.vec_s_pos + self.vec_t_pos
        training_x, testing_x = self.features1[:self.split], self.features1[self.split:]
        # SVM
        clf_SVM_bl = svm.LinearSVC()
        clf_SVM_bl.fit(training_x, self.training_y)
        predictions_baseline_SVM = clf_SVM_bl.predict(testing_x)
        print('*' * 9, 'performance of baseline with svm', '*' * 9)
        self.evaluation(predictions_baseline_SVM)

        # DT
        clf_decision_bl = DecisionTreeClassifier(random_state=2)
        clf_decision_bl.fit(training_x, self.training_y)
        predictions_baseline_DT = clf_decision_bl.predict(testing_x)
        print('*' * 9, 'performance of baseline with Decision Tree', '*' * 9)
        self.evaluation(predictions_baseline_DT)

        return

    def proposed(self):
        '''
        :return: results based on two baseline and proposed feature extraction methods
        '''
        self.proposed_f1()
        self.features2 = np.concatenate((self.features1, self.dis_cos), axis=1)

        training_x, testing_x = self.features2[:self.split], self.features2[self.split:]
        # SVM
        clf_SVM_pro = svm.LinearSVC()
        clf_SVM_pro.fit(training_x, self.training_y)
        predictions_proposed_SVM = clf_SVM_pro.predict(testing_x)
        print('*' * 9, 'performance of proposed features with svm', '*' * 9)
        self.evaluation(predictions_proposed_SVM)

        # DT
        clf_decision_pro = DecisionTreeClassifier(random_state=2)
        clf_decision_pro.fit(training_x, self.training_y)
        predictions_proposed_DT = clf_decision_pro.predict(testing_x)
        print('*' * 9, 'performance of proposed features with Decision Tree', '*' * 9)
        self.evaluation(predictions_proposed_DT)
        return


def main(baseline=1, proposed_method=1):
    s = Solution()
    s.DataLoader()
    if baseline == True:
        s.baseline()
    if proposed_method == True:
        s.proposed()


if __name__ == '__main__':
    main()
