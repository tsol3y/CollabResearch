import numpy as np
import os
import glob
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
from QModel_new import QModel
import csv
# li = formatData(li)

# li = getFiles("")
# dfs = li[:len(li)-6]
# test = li[len(li)-6:]



class MAML(object):
    def __init__(self, session, ld_theta, stocks_sector):
        self.test_called = 0
        self.num_train_samples = 35
        self.num_meta_samples = 35
        self.num_test_samples = 5
        self.num_features = 2
        self.num_outputs = 1
        self.num_layers = 2

        #number of epochs i.e training iterations
        self.epochs = 50
        
        #hyperparameter for the inner loop (inner gradient update)
        self.alpha = 0.001
        
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        self.beta = 0.001
        self.sess = session
        self.save_path = "last_theta.npy"  
        if ld_theta: self.theta = np.load(self.save_path)
        else: self.theta = np.random.normal(size=4*self.num_features-1) # tf.Variable([0.1]*9)
        
        all_files = glob.glob(stocks_sector + "/*.csv")
        dfs = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            dfs.append(df)

        self.dfs_train = dfs[:len(dfs)-6]
        self.dfs_test = dfs[len(dfs)-6:]

    #define our sigmoid activation function 
    def sample_points(self, df,size):                                         # return sample points with two x (features) and one y
        start = random.randint(0,len(df)-size)
        end = start + size - 1
        x1 = df.loc[start:end, ['Close']]
        x2 = df.loc[start:end, ['Volume']]
        y1 = df.loc[start:end, ['ClassLabel']]
        x1 = np.array(x1.values.tolist())
        y1 = np.array(y1.values.tolist())
        x2 = np.array(x2.values.tolist())
        scaler0 = MinMaxScaler()
        scaler0.fit(x1)
        scaler1 = MinMaxScaler()
        scaler1.fit(x2)
        x = np.array([a for a in zip(scaler0.transform(x1), scaler1.transform(x2))])
        y = y1.reshape(size,1)
        return x,y 
    def sigmoid(self, a):
        return 1.0 / (1 + np.exp(-a))
    
    def classify(self, value):
        return 1 if value > 0.5 else 0    

    def saveTheta(self):
        print("Saving theta: ", self.theta)
        np.save(self.save_path,self.theta)

    def loadTheta(self):
        self.theta = np.load(self.save_path)
        print("Loading theta: ", self.theta)
    
    #now let us get to the interesting part i.e training :P
    def train(self, num_tasks):
        if num_tasks <= len(self.dfs_train):
            indices = np.random.choice(len(self.dfs_train), num_tasks, False) 
            #for the number of epochs,
            for e in range(self.epochs):        
                print("Epoch: ", e+1)
                theta_ = []
                print("Training Now ===========================================================")
                

                #for task i in batch of tasks
                for i in range(num_tasks):
                    TrainModel = QModel(self.sess, self.theta, self.num_features, self.num_outputs, self.num_layers)
                    
                    XTrain, YTrain = self.sample_points(self.dfs_train[indices[i]],self.num_train_samples)
                    TrainModel.fit(XTrain,YTrain)
                    gradient = TrainModel.calc_grad(XTrain,YTrain)
                    print(gradient)
                    #update the gradients and find the optimal parameter theta' for each of tasks
                    theta_.append(self.theta - self.alpha*gradient)


                #initialize meta gradients
                meta_gradient = np.zeros(self.theta.shape)

                for i in range(num_tasks):
                    XMeta, YMeta = self.sample_points(self.dfs_train[indices[i]],self.num_meta_samples)
                    print("Meta Now --------------------------------------------- ")
                    QMetaModel = QModel(self.sess,tf.Variable(theta_[i][0]), self.num_features, self.num_outputs, self.num_layers)
                    QMetaModel.fit(XMeta, YMeta)
                    meta_gradient += QMetaModel.calc_grad(XMeta, YMeta)[0]
                #update our randomly initialized model parameter theta with the meta gradients
                self.theta = self.theta-self.beta*meta_gradient/num_tasks
                self.saveTheta()

    def test(self, df, sample_size, test_size):   
        TestModel = QModel(self.sess, self.theta, self.num_features, self.num_outputs, self.num_layers)
        train_start = 0
        train_end = train_start + sample_size
        XTest, YTest = self.sample_points(df,sample_size + test_size)
        TestModel.fit(XTest[:train_end],YTest[:train_end])
        print("Testing Now +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_")
        YPred = TestModel.predict(XTest[train_end:])
        YPred = [self.classify(pred) for pred in YPred]
        
        correct = 0
        for index in range(0,test_size):
            if [YPred[index]] == YTest[train_end+index]: correct += 1
        accuracy = (correct/self.num_test_samples) * 100
        print("Predicted {}".format(YPred))
        print("Actual {}".format(YTest[train_end:]))
        print("Accuracy {}%\n".format(accuracy))
        with open('Results/results{}.csv'.format(self.test_called), 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['YTest','YPred'])
            for i in range (len(YPred)):
                filewriter.writerow([YTest[train_end+i], YPred[i]])
        self.test_called += 1  

model = MAML(session = tf.Session(), ld_theta = False, stocks_sector = "fStocks_Sector_CC")
for i in range(6):
    model.train(num_tasks = 10)
    for j in range(len(model.dfs_test)):
        model.test(df = model.dfs_test[j], sample_size = 20, test_size = 5)