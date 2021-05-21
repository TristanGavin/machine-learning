import numpy as np
import csv
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=.0002
max_iters=800

def main():

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()


  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  X_train_bias = dummyAugment(X_train)
 
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))


  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  logging.info("\n---------------------------------------------------------------------------\n")


  logging.info("Running cross-fold validation for bias case:")


  # Perform k-fold cross
  for k in [2,3,4, 5, 10, 20, 50]:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################

  X_test_bias = np.insert(X_test, 0, np.ones(233), axis=1)
  test_pred = X_test_bias@w >= 0 
  output = [[y, 1 if test_pred[y] else 0] for y in range(len(test_pred))]
  
  with open('submission.csv','w', newline='') as out:
      csv_out = csv.writer(out)
      csv_out.writerow(['id','type'])
      for row in output:
        num, y = row
        csv_out.writerow((num, y))


  raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')



def dummyAugment(X):
  return np.insert(X, 0, np.ones(466), axis=1)


def calculateNegativeLogLikelihood(X,y,w):
  # -sum(ylogsigma(wx) + (1-y)log(1-sigma(wtx))
  x = np.matmul(X,w)
  z = 1/(1 + np.exp(-x))
  negative_log_like = -sum(y*np.log(z+0.0000001)+(1-y)*np.log(1-z+0.00000001))
  return negative_log_like
 

def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):

    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) )
    
    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):
    
        # Make a variable to store our gradient
        w_grad = np.zeros( (X.shape[1],1) )

        # Compute the gradient over the dataset and store in w_grad
        # .
        # . Implement equation 9.
        # .
        w_grad = sum((1/(1 + np.exp(-np.matmul(X,w)))-y)*X)
        w_grad = np.transpose(w_grad[np.newaxis])


        # This is here to make sure your gradient is the right shape
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad
        
        # Calculate the negative log-likelihood with the 
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X,y,w))

    return w, losses




##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()