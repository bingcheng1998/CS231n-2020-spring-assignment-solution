import numpy as np
import matplotlib.pyplot as plt
from cs231n.vis_utils import visualize_grid
import pickle
import os.path

## Data to pickle

def savedb(obj,filename):
    with open(filename,'wb') as file:
        pickle.dump(obj,file)
    
def loaddb(filename):
    with open(filename,'rb') as file:
        obj = pickle.load(file)
        return obj
    
def pickle_exist(hs, bs, lr, reg, num_epoch):
    filename = f'pickle/{hs}-{bs}-{lr}-{reg}-{num_epoch}.pickle'
    if os.path.isfile(filename):
        return True
    return False

def json_exist(hs, bs, lr, reg, num_epoch):
    filename = f'json/{hs}-{bs}-{lr}-{reg}-{num_epoch}.json'
    if os.path.isfile(filename):
        return True
    return False

def save_pickle(hs, bs, lr, reg, num_epoch, val_acc, W1, stats, dtype = np.half):
    W1 = dtype(W1)
    for key in stats.keys():
        stats[key] = dtype(stats[key])
    obj = (hs, bs, lr, reg, num_epoch, val_acc, W1, stats)
    filename = f'pickle/{hs}-{bs}-{lr}-{reg}-{num_epoch}.pickle'
    savedb(obj,filename)
    
def get_pickle(hs, bs, lr, reg, num_epoch):
    filename = f'pickle/{hs}-{bs}-{lr}-{reg}-{num_epoch}.pickle'
    return loaddb(filename)

## Data to graph

def showTraining(stats, dtype = np.half):
    plt.subplot(2, 1, 1)
    plt.plot(dtype(stats['loss_history']))
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(dtype(stats['train_acc_history']), label='train')
    plt.plot(dtype(stats['val_acc_history']), label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def show_net_weights(W1, dtype = np.half):
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2) #- Xs: Data of shape (N, H, W, C)
    W1 = W1.astype(dtype)
    img = visualize_grid(W1, padding=3).astype('uint8')
    plt.imshow(img)
    plt.gca().axis('off')
    plt.show()
    return img