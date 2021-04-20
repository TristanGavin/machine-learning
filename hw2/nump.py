import numpy as np
import math
import csv
from itertools import count, chain
import sys
import random
import copy

cnt = count()
cnt2 = count()
cnt3 = count()

# Load CSVs into dictionaries
with open('train.csv', mode='r') as file:
    reader = csv.reader(file)
    train = {}
    train_y = {}
    first_row = next(reader)
    for row in reader:
        train[next(cnt)] = np.array(list(float(row[i]) for i in range(1,86)))
        train_y[next(cnt2)] = float(row[86])
        
with open('test_pub.csv', mode='r') as file:
    reader = csv.reader(file)
    test = {}
    first_row = next(reader)
    for row in reader:
        test[next(cnt3)] = np.array(list(float(row[i]) for i in range(1,86)))

# takes (dict,dict,int)
def evaluate_model(train, test, train_y, k):
    classified = []
    for key,row in test.items():
        neighbors = get_kneighbors(trainfold_dict, row, k)
        prediction = make_prediction
        classified.append((key, prediction))

# get k nearest neighbors
# takes (dict, row, k)
# need to use broadcasting
def get_kneighbors(train, test_row, k_neighbors):
    distance = []
    X = np.array(list(x for key, x in train.items()))
    y = X-test_row
    for key, row in train.items():
        distance.append((key, np.linalg.norm(row-test_row)))

    sorted_distance = sorted(distance, key=lambda item: item[1])

    neighbors = []
    for k in range(k_neighbors):
        neighbors.append(sorted_distance[k])
    return neighbors

def make_prediction(neighbors, train_y):
    neighbor_output = list(train_y[n[0]] for n in neighbors)
    prediction = max(set(neighbor_output), key=neighbor_output.count)
    return prediction

# Make prediction
neighbors = get_kneighbors(train, test[0], 5)
neighbor_output = list(train_y[n[0]] for n in neighbors)
prediction = max(set(neighbor_output), key=neighbor_output.count)


#K-Fold cross validation

# randomly split train into 4 equal sized sets

# randomly shuffle the ids in train.
train_keys = list(train.keys())
random.shuffle(train_keys)
fold1 = train_keys[:2000]
fold2 = train_keys[2000:4000]
fold3 = train_keys[4000:6000]
fold4 = train_keys[6000:8000]

folds = [fold1,fold2,fold3,fold4]



# for k in range(len(folds)):
#     trainfold = sum(folds[:k], []) + sum(folds[k+1:], [])
#     # train on train fold
#     #convert back to dictionaries
#     trainfold_dict = {}
#     for i in trainfold:
#         trainfold_dict[i] = train[i]
#     testfold_dict = {}
#     for j in folds[k]:
#         testfold_dict[j] = train[j]
#     # now evaluate model on testfold
#     evaluate_model(trainfold_dict, testfold_dict, train_y, k)




# for each set i 
#   combine all subsets except i
#   train on this data
#   evaluate the model on subset i

#compute average performance over these ktrain/evaluation runs.