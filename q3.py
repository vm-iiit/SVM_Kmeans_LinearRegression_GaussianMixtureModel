from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

class Airfoil:

    def __init__(self, iters=30000, alpha=0.0002, reg_param=0.001):
        self.thetas=[]
        self.iters = iters
        self.alpha = alpha
        self.reg_param = reg_param

    def normalize(self, array):
        c = array.shape[1]
        for ind in range(c):
            col = array[:,ind]
            array[:,ind] = (col - col.min())/(col.max() - col.min())
        return array

    def train(self, trainpath):
        train_data = np.genfromtxt(trainpath, delimiter=',')
        # np.random.shuffle(train_data)
        X_part = train_data[:,0:-1]
        Y_part = train_data[:,-1]
        self.y_test = Y_part.copy()
        X_part = self.normalize(X_part)
        X_part = np.hstack((X_part, np.ones((X_part.shape[0], 1))))
        # print(X_part)
        # x_train, self.x_test, y_train, self.y_test = train_test_split(X_part, Y_part, train_size=1)
        x_train = X_part.copy()
        y_train = Y_part.copy()
        self.thetas = np.random.rand(X_part.shape[1], 1)
        # print("thetas")
        # print(self.thetas)
        # print(x_sums.shape)
        samples = x_train.shape[0]
        y_train = y_train.reshape((-1,1))
        oldcost = math.inf
        tolerance = 0.000000001
        xt = x_train.transpose()
        for it in range(self.iters):
            y_temp = np.dot(x_train, self.thetas)
            err = np.subtract(y_temp, y_train)
            cost = ((np.square(err)).sum() + self.reg_param*(np.square(self.thetas).sum()))/(2*samples)
            if(oldcost - cost <= tolerance):
                break
            oldcost = cost
            # print("iter "+str(it+1)+" cost = "+str(cost))
            temp = np.dot(xt, err)
            temp = temp.reshape((-1,1))
            self.thetas *= (1 - (self.alpha*self.reg_param)/samples)
            self.thetas -= self.alpha*temp
    
    def predict(self, testpath):
        test_data = np.genfromtxt(testpath, delimiter=',')
        X_part = test_data
        # Y_part = test_data[:,-1]
        X_part = self.normalize(X_part)
        X_part = np.hstack((X_part, np.ones((X_part.shape[0], 1))))
        preds = np.ndarray.tolist(np.dot(X_part, self.thetas).flatten())
        # print("score")
        # print(r2_score(self.y_test, preds))
        # print(preds)
        return preds

