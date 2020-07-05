import numpy as np
import pandas as pd
import global_align as ga
import torch

db_name = "ECGFiveDays"
xx = pd.read_csv("/Users/profloolab/Desktop/Hemin/After_PD/db/UCRArchive_2018/{}/{}_TRAIN.tsv".format(db_name,db_name), sep="\t", header=None, index_col=None)
te_x = pd.read_csv("/Users/profloolab/Desktop/Hemin/After_PD/db/UCRArchive_2018/{}/{}_TEST.tsv".format(db_name,db_name), sep="\t", header=None, index_col=None)


xx = np.array(xx)
te_x = np.array(te_x)

train_x = xx[:,1:]
train_y = xx[:,0]
test_x = te_x[:,1:]
test_y = te_x[:,0].astype(np.int64)

from sklearn import preprocessing
test_x = preprocessing.scale(test_x)
train_x = preprocessing.scale(train_x)


train_x = torch.from_numpy(train_x)
test_x = torch.from_numpy(test_x)
power=2
norm = train_x.pow(power).sum(1, keepdim=True).pow(1./power)
train_x = train_x.div(norm)
norm2 = test_x.pow(power).sum(1, keepdim=True).pow(1./power)
test_x = test_x.div(norm2)


train_set = np.concatenate((train_x, train_y.reshape(len(train_y),1)), 1)
test_set = np.concatenate((test_x, test_y.reshape(len(test_y),1)), 1)
# top = 0.
C = train_y.max()
retrieval_one_hot = torch.zeros(1, C)
trainLabels = torch.from_numpy(train_y).long()

class KNearestNeighbors(object):
    def __init__(self, k, sig, top):
        self.k = k
        self.sig = sig
        self.top = 0.
    @staticmethod
    def _euclidean_distance(v1,v2):
        v1, v2 = np.array(v1), np.array(v2)
        distance = 0
        for i in range(len(v1) - 1):
            distance += (v1[i] - v2[i]) ** 2
        return np.sqrt(distance)
    def _GA(self, v1, v2):
        sq1 = v1.reshape(1, len(v1)).astype('double')
        sq2 = v2.reshape(1, len(v2)).astype('double')
        myv = ga.tga_dissimilarity(sq1, sq2, self.sig, 0)
        myv2 = np.exp(-myv)
        sim = 1 - myv2
        return sim
    def predict(self, train_set, test_instance):
        distances = []
        for i in range(len(train_set)):
            dist = self._euclidean_distance(train_set[i][:-1], test_instance)
            distances.append((train_set[i], dist))
        distances.sort(key=lambda x: x[1])

        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        classes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1
        sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        sss = sorted_classes[0][0]
        return sorted_classes[0][0]

    def predict_ga(self, train_set, test_instance):
        distances = []
        for i in range(len(train_set)):
            dist = self._GA(train_set[i][:-1], test_instance)
            distances.append((train_set[i], dist))
        distances.sort(key=lambda x: x[1])

        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        classes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1
        sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        ss = sorted_classes[0][0]
        return sorted_classes[0][0]




    @staticmethod
    def evaluate(y_true, y_pred):
        n_correct = 0
        for act, pred in zip(y_true, y_pred):
            if act == pred:
                n_correct += 1
        ff = float(n_correct) / float(len(y_true))
        return ff
for k in range(1,2,2):

    knn = KNearestNeighbors(k=3,sig=654.333, top=None)
    preds = []
    preds_ga = []
    for row in test_set:
        predictors_only = row[:-1]
        target = row[len(predictors_only)]
        prediction = knn.predict(train_set, predictors_only)
        prediction_ga = knn.predict_ga(train_set, predictors_only)


        preds.append(prediction)
        preds_ga.append(prediction_ga)


    actual = np.array(test_set)[:, -1]
    print("k={} , Acc Eu= {} , GA= {} ".format(k,knn.evaluate(actual, preds)*100, knn.evaluate(actual, preds_ga)*100))
