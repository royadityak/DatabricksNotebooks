
# coding: utf-8

# In[41]:


import random
import numpy as np
from collections import namedtuple, Counter, defaultdict
import gzip
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


# In[42]:


Dataset = namedtuple('Dataset', ['inputs', 'labels'])

# Reading in data. You do not need to touch this.
with open("../data/train-images-idx3-ubyte.gz", 'rb') as f1, open("../data/train-labels-idx1-ubyte.gz", 'rb') as f2:
    buf1 = gzip.GzipFile(fileobj=f1).read(16 + 60000 * 28 * 28)
    buf2 = gzip.GzipFile(fileobj=f2).read(8 + 60000)
    inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(60000, 28 * 28)[0:5000,:]
    inputs = np.where(inputs > 99, 1, 0)
    labels = np.frombuffer(buf2, dtype='uint8', offset=8)[0:5000,]
    data_train = Dataset(inputs, labels)
    data_train_test = Dataset(inputs[10000:40000,], labels[10000:40000,])
    
with open("../data/t10k-images-idx3-ubyte.gz", 'rb') as f1, open("../data/t10k-labels-idx1-ubyte.gz", 'rb') as f2:
    buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
    buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
    inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(10000, 28 * 28)
    inputs = np.where(inputs > 99, 1, 0)[150:600,:]
    labels = np.frombuffer(buf2, dtype='uint8', offset=8)[150:600,]
    data_test = Dataset(inputs, labels)


# In[43]:


def calculate_accuracy (labels, pred):
    score = 0
    right_assignment = len([1 for itr in range(len(pred)) if pred[itr] == labels[itr]])
    return (right_assignment / float(len(labels)))*100.0


# In[50]:


#Gaussian Naive Bayes
gnb = GaussianNB().fit(data_train.inputs, data_train.labels)
print ("Accuracy on Test = " + str(calculate_accuracy(data_test.labels, gnb.predict(data_test.inputs))))
print ("Accuracy on Training = " + str(calculate_accuracy(data_train_test.labels, gnb.predict(data_train_test.inputs))))


# In[51]:


# Multinomial Naive Bayes
mnb = MultinomialNB().fit(data_train.inputs, data_train.labels)
y_pred = mnb.fit(data_train.inputs, data_train.labels).predict(data_test.inputs)
accuracy = calculate_accuracy (data_test.labels, y_pred)

print ("Accuracy on Test = " + str(calculate_accuracy(data_test.labels, mnb.predict(data_test.inputs))))
print ("Accuracy on Training = " + str(calculate_accuracy(data_train_test.labels, mnb.predict(data_train_test.inputs))))


# In[46]:


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# In[47]:


class LogisticRegression:
    '''
    Multinomial Linear Regression that learns weights by minimizing
    mean squared error using stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_features, n_classes))  # An extra row added for the bias
        self.alpha = 0.0025  # tune this parameter

    def train(self, X, Y):
        '''
        Trains the model, using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None. You can change this to return whatever you want, e.g. an array of loss
            values, to produce data for your project report.
        '''
        flag = 1
        random_matrix = [i for i in range(len(Y))]

        while (flag):
            np.random.shuffle(random_matrix)

            for itr in random_matrix:
                classifier_ls = np.dot(X[itr].T, self.weights)
                probs = np.apply_along_axis(softmax, -1, classifier_ls)

                delta_j = np.zeros(self.n_classes)
                for j in range(self.n_classes):
                    if Y[itr] == j:
                        delta_j[j] = probs[j] - 1
                    else:
                        delta_j[j] = probs[j]

                delta_j = delta_j.reshape(-1, 1)
                delta_w = np.dot(X[itr].reshape(-1, 1), delta_j.T)

                self.weights -= (self.alpha * delta_w)

                if (np.allclose((self.alpha * delta_w), 0, rtol=0.00000001, equal_nan=True)):
                    flag = 0
                    break

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        predictions = np.zeros(len(X))
        for i in range(len(predictions)):
            l = np.dot(X[i], self.weights)
            #p = softmax(l)
            p = np.apply_along_axis(softmax, -1, l)
            predictions[i] = np.argmax(p)
        return predictions

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        return np.mean(np.equal(self.predict(X), Y))


# In[48]:


#X_train, Y_train, X_test, Y_test = import_mnist(MNIST_TRAIN_INPUTS_PATH, MNIST_TRAIN_LABELS_PATH, MNIST_TEST_INPUTS_PATH, MNIST_TEST_LABELS_PATH)
num_features = data_train.inputs.shape[1]

print('--------- LOGISTIC REGRESSION w/ SGD ---------')
model = LogisticRegression(num_features, 10)
model.train(data_train.inputs, data_train.labels)
print("Test Accuracy: {:.1f}%".format(model.accuracy(data_test.inputs, data_test.labels) * 100))
print("Training Accuracy: {:.1f}%".format(model.accuracy(data_train_test.inputs, data_train_test.labels) * 100))

