import numpy as np
import math
import csv
from itertools import count, chain
import sys
import random
import copy
import time

cnt = count()
cnt2 = count()
cnt3 = count()

# Load CSVs into dictionaries
with open('train.csv', mode='r') as file:
    reader = csv.reader(file)
    global_train = {}
    train_y = {}
    first_row = next(reader)
    for row in reader:
        global_train[next(cnt)] = np.array(list(float(row[i]) for i in range(1,86)))
        train_y[next(cnt2)] = float(row[86])
        
with open('test_pub.csv', mode='r') as file:
    reader = csv.reader(file)
    global_test = {}
    first_row = next(reader)
    for row in reader:
        global_test[next(cnt3)] = np.array(list(float(row[i]) for i in range(1,86)))

# takes (list(keys),list(keys), dict, int)
def evaluate_model(train, test, train_y, k):
    classified = []
    accuracy = 0
    for row in test:
        neighbors = get_kneighbors(train, row, k)
        prediction = make_prediction(neighbors, train_y)
        if train_y[row] == prediction:
            accuracy += 1
    print(accuracy/len(test))

# get k nearest neighbors
# takes (list(keys), int(key) to a row, k)
# need to use broadcasting
def get_kneighbors(train, test_row, k_neighbors):
    x = np.array(list(global_train[key] for key in train))
    y = x-global_train[test_row]
    ynorm = np.linalg.norm(y, axis=1)
    distance = list((train[i], ynorm[i]) for i in range(len(train)))
    sorted_distance = sorted(distance, key=lambda item: item[1])

    neighbors = []
    for k in range(k_neighbors):
        neighbors.append(sorted_distance[k])
    return neighbors

def make_prediction(neighbors, train_y):
    neighbor_output = list(train_y[n[0]] for n in neighbors)
    prediction = max(set(neighbor_output), key=neighbor_output.count)
    return prediction

# # Make prediction
# neighbors = get_kneighbors(train, test[0], 5)
# neighbor_output = list(train_y[n[0]] for n in neighbors)
# prediction = max(set(neighbor_output), key=neighbor_output.count)

def train_on_train(train_keys,k):
    evaluate_model(train_keys, train_keys, train_y, k)


#K-Fold cross validation
# randomly split train into 4 equal sized sets (randomly shuffle the ids in train)
train_keys = list(global_train.keys())

def k_fold(train_keys):
    random.shuffle(train_keys)
    fold1 = train_keys[:2000]
    fold2 = train_keys[2000:4000]
    fold3 = train_keys[4000:6000]
    fold4 = train_keys[6000:8000]

    folds = [fold1,fold2,fold3,fold4]

    for k in range(len(folds)):
        trainfold = sum(folds[:k], []) + sum(folds[k+1:], [])
        testfold = folds[k]
        # now evaluate model on testfold
        evaluate_model(trainfold, testfold, train_y, 5)

start_time = time.time()
train_on_train(train_keys,999)
print("--- %s seconds ---" % (time.time() - start_time))