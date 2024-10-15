from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from RareTransfer import rareTransfer
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score


class RUN(object):
    def __init__(self, Xs, Ys,Sw, Xt, Yt, n_neighbors=10, iter=20, clf='Boost',
                 n_estimators=10, criterion='gini', max_features='auto', RFmin_samples_split=2,     # RF
                 Boostnestimator=50, BoostLearnrate=0.99,                                              # Boost
                 CARTsplitter='best',                                                               # CART
                 Ridgealpha=1, Ridgenormalize=False,                                                # Ridge
                 NBtype='bernoulli', #multinomial bernoulli
                 SVCkernel='poly', C=1, degree=3, coef0=0.0, SVCgamma=1
                 ):
        self.Xsource = np.asarray(Xs)
        self.Ysource = np.asarray(Ys)
        self.source_weight=np.asarray(Sw)
        self.Xtarget = np.asarray(Xt)
        self.Ytarget = np.asarray(Yt)
        self.n_neighbors = int(n_neighbors)
        self.iter = iter
        self.clfType = clf
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.RFmin_samples = RFmin_samples_split
        self.Boostne = Boostnestimator
        self.BoostLearnrate = BoostLearnrate
        self.NBType = NBtype
        self.CARTsplitter = CARTsplitter
        self.Ridgealpha = Ridgealpha
        self.Ridgenormalize = Ridgenormalize
        self.SVCkernel = SVCkernel
        self.coef0 = coef0
        self.gamma = SVCgamma
        self.degree = degree
        self.C = C


    def _NNfilter(self):
        knn = NearestNeighbors()
        knn.fit(self.Xsource) 
        data = []
        ysel = []
        wdata=[]

        for item in self.Xtarget:
            tmp = knn.kneighbors(item.reshape(1, -1), self.n_neighbors, return_distance=False)
            
            tmp = tmp[0]
            for i in tmp:
                if list(self.Xsource[i]) not in data:
                    data.append(list(self.Xsource[i]))
                    
                    ysel.append(self.Ysource[i])
        self.Xsource = np.asanyarray(data)
        # print(self.Xsource.shape)
        self.source_weight=np.asarray(wdata)
        # print(self.source_weight.shape,'*')
        self.Ysource = np.asanyarray(ysel)
        # print(self.Ysource.shape)


    def _SMOTE(self):
        smote = SMOTE()

        self.Xsource, self.Ysource = smote.fit_resample(self.Xsource, self.Ysource)

    def _max_min(self, x):
        shape = np.asarray(x).shape
        Max = np.zeros(shape[1])
        Min = np.zeros(shape[1])
        for i in range(0, shape[1]):
            a = x[:, i]
            Max[i] = np.max(a)
            Min[i] = np.min(a)

        return Max, Min


    def _weight(self):
        max, min = self._max_min(self.Xtarget)
        shape = self.Xsource.shape
        # print(shape)
        s = np.zeros(shape[0])
        w = np.zeros(shape[0])
        for i in range(0,shape[0]):
            tmp = 0
            for j in range(0, shape[1]):
                if self.Xsource[i][j] <= max[j] and self.Xsource[i][j] >= min[j]:
                    tmp = tmp + 1
            s[i] = tmp


            w[i] = s[i] / (1.0 * np.power(shape[1] - s[i] + 1, 2))
        # print(w)
        return w



    def fit(self):

        self._NNfilter()
        self._SMOTE()
        weight = self._weight()

        trainX, self.testX, trainY, self.testY = train_test_split(self.Xtarget, self.Ytarget, test_size=0.90)
        while len(np.unique(self.testY)) <= 1:
            trainX, self.testX, trainY, self.testX = train_test_split(self.Xtarget, self.Ytarget, test_size=0.90)

        if self.clfType == 'RF':
            m = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                            max_features=self.max_features, min_samples_split=self.RFmin_samples)
        if self.clfType == 'SVM':
            m = SVC(kernel=self.SVCkernel, C=self.C, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        if self.clfType == 'Boost':
            m = AdaBoostClassifier(n_estimators=self.Boostne, learning_rate=self.BoostLearnrate)
        if self.clfType == 'NB':
            if self.NBType == 'gaussian':
                m = GaussianNB()
            elif self.NBType == 'multinomial':
                m = MultinomialNB()
            elif self.NBType == 'bernoulli':
                m = BernoulliNB()
        if self.clfType == 'CART':
            m = DecisionTreeClassifier(criterion=self.criterion, splitter=self.CARTsplitter, max_features=self.max_features, min_samples_split=self.RFmin_samples)
        if self.clfType == 'Ridge':
            m = RidgeClassifier(alpha=self.Ridgealpha, normalize=self.Ridgenormalize)

        self.model = rareTransfer(trainX, self.Xsource, trainY, self.Ysource, self.testX, self.testY, self.iter, initWeight=weight / (weight * (self.Xsource.shape[0] + trainX.shape[0])), clf=m)
        #np.savez('data.npz', trainX = trainX , Xsource = self.Xsource , trainY = trainY , Ysource = self.Ysource , testX = self.testX , testY = self.testY , weight = weight)

        self.model.fit()


    def predict(self):

        
        Ypredict, testY= self.model.predict()

        

        self.AUC = roc_auc_score(self.testY, Ypredict)
        self.F=f1_score(self.testY, Ypredict)
        self.Accuracy = accuracy_score(self.testY,Ypredict)

    def predict_single_instance(self, instance):


        Ypredict = self.model.predict_single_instance(instance=instance)

        return Ypredict
    
    
    




