import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt


get_ipython().run_line_magic("matplotlib", " inline")
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


from cs231n.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()


from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def tryParams(lr, reg, best_val, best_svm):
    svm = LinearSVM()
#     print(f"lr = {lr}, reg = {reg}:")
    loss_hist = svm.train(X_train_feats, y_train, learning_rate=1e-7, reg=2.5e4,
                          num_iters=5000, verbose=False)
    y_train_pred = np.mean(y_train == svm.predict(X_train_feats))
    y_val_pred = np.mean(y_val == svm.predict(X_val_feats))
    results[(lr, reg)] = (y_train_pred,y_val_pred)
    if y_val_pred > best_val: 
        best_val = y_val_pred
        best_svm = svm
    return best_val, best_svm
    
for lr in learning_rates:
    for reg in regularization_strengths:
        best_val, best_svm = tryParams(lr, reg, best_val, best_svm)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr get_ipython().run_line_magic("e", " reg %e train accuracy: %f val accuracy: %f' % (")
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: get_ipython().run_line_magic("f'", " % best_val)")


# Evaluate your trained SVM on the test set: you should be able to get at least 0.40
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)


# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test get_ipython().getoutput("= cls) & (y_test_pred == cls))[0]")
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()


# Preprocessing: Remove the bias dimension
# Make sure to run this cell only ONCE
print(X_train_feats.shape)
X_train_feats = X_train_feats[:, :-1]
X_val_feats = X_val_feats[:, :-1]
X_test_feats = X_test_feats[:, :-1]

print(X_train_feats.shape)


get_ipython().run_cell_magic("time", "", """from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

state = None
input_size = input_dim
num_classes = 10
best_acc = 0

hidden_size = [hidden_dim]
batch_size = [200, 300, 400]
learning_rate = [1e-2 ,1e-1, 5e-1, 1, 2]
regularization = [1e-3, 5e-3, 1e-2]

best_hs = hidden_size[0]
best_bs = batch_size[1]
best_lr = learning_rate[2]
best_reg = regularization[1]

def testParam(hs, bs, lr, reg):
    net = TwoLayerNet(input_size, hs, num_classes)
    num_train = X_train_feats.shape[0]
    iterations_per_epoch = max(num_train / bs, 1)
    num_epoch = 10
    num_iters = int(iterations_per_epoch*num_epoch)
    # Here exactly 5 epoch are runned for each parameter set
    stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
                num_iters=num_iters, batch_size=bs,
                learning_rate=lr, learning_rate_decay=0.95,
                reg=reg, verbose=False)
    val_acc = (net.predict(X_val_feats) == y_val).mean()
    print(f'hs: {hs}, bs: {bs}, lr: {lr}, reg:{reg}, val_acc: {val_acc}')
    return val_acc, net, stats

        
for lr in learning_rate:
    val_acc, net, stats = testParam(best_hs, best_bs, lr, best_reg)
    if val_acc > best_acc:
        best_acc = val_acc
        best_lr = lr
        best_net = net
        
for hs in hidden_size:
    val_acc, net, stats = testParam(hs, best_bs, best_lr, best_reg)
    if val_acc > best_acc:
        best_acc = val_acc
        best_hs = hs
        best_net = net

for bs in batch_size:
    val_acc, net, stats = testParam(best_hs, bs, best_lr, best_reg)
    if val_acc > best_acc:
        best_acc = val_acc
        best_bs = bs
        best_net = net

        
for reg in regularization:
    val_acc, net, state = testParam(best_hs, best_bs, best_lr, reg)
    if val_acc > best_acc:
        best_acc = val_acc
        best_reg = reg
        best_net = net
        
print(f'best_hs:{best_hs}, best_bs:{best_bs}, best_lr:{best_lr}, best_reg:{best_reg}, best_acc:{best_acc}')    

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
""")


def showTraining(stats):
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.show()


showTraining(state)
# show_net_weights(best_net)


# Run your best neural net classifier on the test set. You should be able
# to get more than 55% accuracy.

test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print(test_acc)



