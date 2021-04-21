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
# loop through each row in test rows and get nearest neighbors, store accuracy. (only used when training model)
def evaluate_model(train, test, train_y, k, t_on_t):
    classified = []
    accuracy = 0
    for row in test:
        neighbors = get_kneighbors(train, row, k, t_on_t)
        prediction = make_prediction(neighbors, train_y)
        if train_y[row] == prediction:
            accuracy += 1
    print(accuracy/len(test))

# get k nearest neighbors
# takes (list(keys), int(key) to a row, k)
def get_kneighbors(train, test_row, k_neighbors, t_on_t):
    x = np.array(list(global_train[key] for key in train))
    #if training use x-train data otherwise find x-test 
    if t_on_t == True:
        y = x-global_train[test_row]
    else:
        y = x-global_test[test_row]

    #calculate norm (use broadcasting to do whole row at a time)
    ynorm = np.linalg.norm(y, axis=1)
    distance = list((train[i], ynorm[i]) for i in range(len(train)))
    sorted_distance = sorted(distance, key=lambda item: item[1])

    # create array of the k shortest distances
    neighbors = []
    for k in range(k_neighbors):
        neighbors.append(sorted_distance[k])
    return neighbors

# take max of the k-nearest neighbor outputs and make that the prediction for this data point
def make_prediction(neighbors, train_y):
    neighbor_output = list(train_y[n[0]] for n in neighbors)
    prediction = max(set(neighbor_output), key=neighbor_output.count)
    return prediction

# function to run if evaluating test data
def test_evaluation(train, test, k):
    t_on_t = False # poorly named variable to tell program wether this is new test data, or if we are training.
    prediction_list = []
    for row in test:
        neighbors = get_kneighbors(train, row, k, t_on_t)
        prediction = make_prediction(neighbors, train_y)
        prediction_list.append((int(row), int(prediction)))
    return prediction_list

# function to run to train on train.
def train_on_train(train_keys,k):
    t_on_t = True
    evaluate_model(train_keys, train_keys, train_y, k, t_on_t)

# function to run if doing k_fold validation
def k_fold(train_keys, kn):
    t_on_t = True
    # randomly split train into 4 equal sized sets (randomly shuffle the ids in train)
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
        evaluate_model(trainfold, testfold, train_y, kn, t_on_t)

start_time = time.time()

train_keys = list(global_train.keys())
test_keys = list(global_test.keys())

# k_fold(train_keys, 6000)
# train_on_train(train_keys,8000)
output = test_evaluation(train_keys, test_keys, 72)
with open('submission.csv','w', newline='') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['id','income'])
    for row in output:
        csv_out.writerow(row)

print("--- %s seconds ---" % (time.time() - start_time)) # just for tracking run time