import numpy as np
import pandas as pd

# H   Test sample classification results
# TrainS  original training sample np array
# TrainA Auxiliary training samples (the labeled diff-distribution data are treated as the auxiliary data)
# LabelS  Original training sample labels
# LabelA  Auxiliary training sample labels
# Test   test sample
# N  number of iterations


class rareTransfer(object):
    def __init__(self, train_WC,  train_CC, label_WC, label_CC, testX, testY, N, initWeight, clf):
        self.train_WC = train_WC #target training
        self.train_CC = train_CC #source
        self.label_WC = label_WC
        self.label_CC = label_CC
        self.N = N
        self.test = testX
        self.test_l = testY
        self.weight = initWeight
        self.m = clf
        self.error = 0



    def fit(self ):

        train_data = np.concatenate((self.train_CC, self.train_WC), axis=0)
        train_label = np.concatenate((self.label_CC, self.label_WC), axis=0)

        row_CC = self.train_CC.shape[0]
        row_WC = self.train_WC.shape[0]
        row_Test = self.test.shape[0]
        N = self.N

        test_data = np.concatenate((train_data, self.test), axis=0)

        # Initialize weights
        weights_CC = self.weight.reshape(-1, 1)
        weights_WC = np.ones([row_WC, 1])*self.train_WC.shape[1]

        weights = np.concatenate((weights_CC, weights_WC), axis=0)



        # prevent division by zero
        if N == 0 or (1 + np.sqrt(2 * np.log(row_CC / N))) == 0:
            self.error = 1
            return
        beta = 1 / (1 + np.sqrt(np.log(2 * row_CC)/ N))

        #  Storing labels and beta values for each iteration?
        beta_T = np.zeros([1, N])
        result_label = np.ones([row_CC + row_WC + row_Test, N])

        predict = np.zeros([row_Test])

        # print('params initial finished.')
        train_data = np.asarray(train_data, order='C')
        train_label = np.asarray(train_label, order='C')
        test_data = np.asarray(test_data, order='C')
        
        #input_data = [2, 5, 2, 1, 3, 8, 3, 0.21, 4, 2, 1, 0.33, 12, 1, 0, 13, 1, 17.5, 12.5, 2734.06,0.07, 46, 0.08, 151.89, 218.72, 0.33, 2, 4, 11, 0.15, 20, 26, 12, 15, 20, 23.53, 14]
        input_data = [19, 17, 13, 0, 0, 4, 15, 0.27, 2, 2, 14, 0.93, 35, 1, 0, 55, 1, 48.69, 19.67, 18831.94, 0.32, 159, 0.05, 1046.22, 957.56, 0.07, 1, 2, 22, 0.2, 59, 100, 39, 26, 75, 0, 55]
        instance = np.array(input_data).reshape(1, -1)
        

        for i in range(N):

            weights_WC= weights[row_CC:row_CC + row_WC, :] / np.sum(weights)
            weights_CC = weights[0:row_CC, :] / np.sum(weights)

            weights = np.concatenate((weights_CC, weights_WC), axis=0)

            P = self.calculate_P(weights, train_label)

            

            result_label[:, i] = self.train_classify(train_data, train_label, test_data, P)

            error_rate_source = self.calculate_error_rate(self.label_CC, result_label[0:row_CC, i], weights[0:row_CC, :])

            error_rate_target = self.calculate_error_rate(self.label_WC, result_label[row_CC:row_CC + row_WC, i], weights[row_CC:row_CC + row_WC, :])



            Cl = 1 - error_rate_source #label-dependent cost factor

            if error_rate_target >= 0.5:
                error_rate_target = 0.5
            beta_T[0, i] =  error_rate_target/(1 - error_rate_target)

            #  Adjust original sample weights (target)
            for j in range(row_WC):
                weights[row_CC + j] = weights[row_CC + j] * np.power(beta_T[0, i],-1*np.abs(result_label[row_CC + j, i] - self.label_WC[j]))


            #  Adjust the sample weight of the auxiliary domain
            for j in range(row_CC):
                weights[j] = Cl*weights[j] * np.power(beta, np.abs(result_label[j, i] - self.label_CC[j]))

        
        #np.savez('data_arrays.npz', train_data=train_data, train_label=train_label, P=P)
        predictionOutput = self.train_classify(train_data, train_label, instance, P)


        

        
        print("Predicted output"+str(predictionOutput))

        for i in range(row_Test):
            # skip labels for training data
            left = np.sum(
                result_label[row_CC + row_WC + i, int(np.ceil(N / 2)):N] * np.log(1 / beta_T[0, int(np.ceil(N / 2)):N]))
            right = 0.5 * np.sum(np.log(1 / beta_T[0, int(np.ceil(N / 2)):N]))


            if left > right or left == right:
                predict[i] = 1
            else:
                predict[i] = 0

        self.label_p = predict

    def predict(self):
        return self.label_p, self.test_l
    
    def predict_single_instance(self, instance):
        # Reshape the single instance into a 2D array with one row
        instance = np.array(instance).reshape(1, -1)

        sample_weight = np.ones((instance.shape[0],))

        # Predict the output using the model
        prediction = 0 #self.train_classify(self.train_CC, self.label_CC, instance, sample_weight)

        # Return the prediction
        return prediction


    def calculate_P(self, weights, label):
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')


    def train_classify(self, train_data, train_label, test_data, P):
        train_data[train_data!=train_data] = 0
        train_label[train_label!=train_label] = 0
        test_data[test_data!=test_data] = 0
        P[P!=P] = 0

        self.m.fit(train_data, train_label, sample_weight=P[:, 0])

        return self.m.predict(test_data)


    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)

        return np.sum(weight[:, 0]*np.abs(label_R - label_H)/ total)
