from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class  AuthorClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(strip_accents='ascii', ngram_range=(1,2), norm='l2', sublinear_tf=True)
        self.classifier = LinearSVC(multi_class='ovr', max_iter=1500)
        self.x_test = []
        self.y_test = []

    def train(self, trainpath):
        df = pd.read_csv(trainpath, index_col=0)
        arr = df.to_numpy().reshape((-1,2))
        # x_train, self.x_test, y_train, self.y_test = train_test_split(arr[:,0], arr[:,-1], train_size=.75)
        x_train = arr[:,0].copy()
        y_train = arr[:,-1].copy()
        self.y_test = y_train
        x_train = self.vectorizer.fit_transform(x_train)
        # print(x_train.shape)
        self.classifier.fit(x_train, y_train)
        # print("trained")
        
    def predict(self, testpath):
        df = pd.read_csv(testpath, index_col=0)
        df = df['text']
        # print(df)
        # arr = df.to_numpy().reshape((-1,1))
        # self.x_test = arr[:,0].copy()
        # y_test = arr[:,-1]
        self.x_test = self.vectorizer.transform(df)
        # print(accuracy_score(self.y_test, self.classifier.predict(self.x_test))*100)
        lis = self.classifier.predict(self.x_test)
        # print(lis)
        return lis

