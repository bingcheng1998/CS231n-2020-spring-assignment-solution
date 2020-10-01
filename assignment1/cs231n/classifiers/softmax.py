from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores) # to avoid numeric instability
        # Otherwise:  RuntimeWarning: invalid value encountered in true_divide
        correct_class_score = scores[y[i]]
        # loss += -np.log(np.exp(correct_class_score)/np.sum(np.exp(scores)))
        loss += -correct_class_score + np.log(np.sum(np.exp(scores)))


        for j in range(num_classes):
            p = np.exp(scores[j])/np.sum(np.exp(scores))
            if j == y[i]:
                dW[:,j] += (p-1)*X[i].T
            else:
                dW[:,j] += p*X[i].T

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2* reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores = X@W
    scores = (scores.T - np.max(scores, axis = 1)).T # to avoid numeric instability
    correct4rows = scores[range(len(y)),y]
    loss = np.sum(-correct4rows + np.log(np.sum(np.exp(scores), axis = 1)))
    dscore = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1,1)
    dscore[range(len(y)),y] -= 1
    dW = X.T@dscore
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2* reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
