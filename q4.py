from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class Weather:

    def __init__(self, iters=1000, alpha=0.000004, reg_param=1):
        self.thetas=[]
        self.iters = iters
        self.alpha = alpha
        self.reg_param = reg_param

    def clean(self, dfpath):
        df = pd.read_csv(dfpath)
        df.drop(columns=['Formatted Date', 'Daily Summary'], inplace=True)
        of = df['Apparent Temperature (C)'].to_numpy().copy()
        df.drop(columns=['Apparent Temperature (C)'], inplace=True)
        df.fillna(value="nothing", inplace=True)
        df = pd.get_dummies(df, columns=['Summary', 'Precip Type'])
        df['Apparent Temperature (C)'] = of
        return df.to_numpy().copy()

    def clean_test(self, dfpath):
        df = pd.read_csv(dfpath)
        df.drop(columns=['Formatted Date', 'Daily Summary'], inplace=True)
        # of = df['Apparent Temperature (C)'].to_numpy().copy()
        # df.drop(columns=['Apparent Temperature (C)'], inplace=True)
        df.fillna(value="nothing", inplace=True)
        df = pd.get_dummies(df, columns=['Summary', 'Precip Type'])
        # df['Apparent Temperature (C)'] = of
        return df.to_numpy().copy()


    def normalize(self, array):
        c = array.shape[1]
        for ind in range(c):
            col = array[:,ind]
            array[:,ind] = (col - col.min())/(col.max() - col.min())
        # print("fin arr")
        # print(array)
        return array

    def train(self, trainpath):
        train_data = self.clean(trainpath)
        # np.random.shuffle(train_data)
        X_part = train_data[:,0:-1].copy()
        # print("train after clean")
        # print(X_part.shape)
        Y_part = train_data[:,-1].copy()
        self.y_test = Y_part
        X_part = self.normalize(X_part)
        X_part = np.hstack((X_part, np.ones((X_part.shape[0], 1))))
        # print("final train")
        # print(X_part.shape)
        # print(X_part)
        x_train = X_part.copy()
        y_train = Y_part.copy()
        # x_train, self.x_test, y_train, self.y_test = train_test_split(X_part, Y_part, train_size=0.7)
        self.thetas = np.random.rand(x_train.shape[1], 1)
        samples = x_train.shape[0]
        y_train = y_train.reshape((-1,1))
        xt = x_train.transpose()
        for it in range(self.iters):
            y_temp = np.dot(x_train, self.thetas)
            err = np.subtract(y_temp, y_train)
            cost = ((np.square(err)).sum() + self.reg_param*(np.square(self.thetas).sum()))/(2*samples)
            # print("iter "+str(it+1)+" cost = "+str(cost))
            temp = np.dot(xt, err)
            temp = temp.reshape((-1,1))
            self.thetas *= (1 - (self.alpha*self.reg_param)/samples)
            self.thetas -= self.alpha*temp

    
    def predict(self, testpath):
        # test_data = np.genfromtxt(testpath, delimiter=',')
        test_data = self.clean_test(testpath)
        # print("test after clean")
        # print(test_data.shape)
        X_part = test_data.copy()
        # Y_part = test_data[:,-1].copy()
        X_part = self.normalize(X_part)
        # print("b4")
        # print(X_part.shape)
        X_part = np.hstack((X_part, np.ones((X_part.shape[0], 1))))
        # print("final")
        # print(X_part.shape)
        self.x_test = X_part.copy()
        preds = np.ndarray.tolist((np.dot(self.x_test, self.thetas)).flatten())
        # print("score")
        # print(preds)
        # print(r2_score(self.y_test, preds))
        return preds