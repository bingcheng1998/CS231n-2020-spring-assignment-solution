import numpy as np


a = np.array([[1,2],[3,4]])
a


b = np.array([[1,0],[0,1],[1,1]])
b


e = np.reshape(b,(3,1,2))
e


c = np.tile(a,(3,1))
# c


d = np.reshape(c,(3,2,2))
d


f = d-e
f


np.sum(np.abs(f),axis = 2)


def D3Broadcast(a, b):
    # a is Axn, b is Bxn
    # Return BxA
    assert b.shape[1] == a.shape[1]
    b_new = np.reshape(b,(b.shape[0],1,b.shape[1]))
    a_new = np.tile(a,(b.shape[0],1))
    a_new = np.reshape(a_new,(b.shape[0],a.shape[0],b.shape[1]))
    f = a_new-b_new
    return np.sum(np.square(f),axis = 2)


D3Broadcast(a,b)


D3Broadcast(b,a)


a_l = 6 
b_l = 3
n = 10
a = np.random.randint(-9,9,a_l*n, dtype=np.int8).reshape(a_l,n)
a.shape


b = np.random.randint(-9,9,b_l*n, dtype=np.int8).reshape(b_l,n)
b.shape


c = np.sum(np.square(a.reshape(a.shape[0],1,a.shape[1])-b),axis = 2)
c.shape
' DO NOT RUN THIS, MEMORY EXPLOISION'
c


# del c
m = np.zeros((a.shape[0],b.shape[0],a.shape[1]))
np.subtract(a.reshape(a.shape[0],1,a.shape[1]), b, m)
ans = np.sum(np.square(m),axis = 2)
ans.shape
ans


a1 = np.sum(np.square(a),axis=1)
a1


b1 = np.transpose(np.sum(np.square(b),axis=1))
b1.transpose()
b1


a2b2 = a1[:,np.newaxis]+b1
a2b2


ba = np.dot(a, b.T)
ba


(a2b2-2*ba)


(-2*np.dot(a, b.T) + 
        np.sum(np.square(b), axis = 1) + 
        np.transpose([np.sum(np.square(a), axis = 1)]))[:,np.newaxis]


a = np.random.randint(-9,9,10, dtype=np.int8)
a_l = np.array_split(a, 4)
a_l


a_l. 


import numpy as np
scores = np.random.random((5,3))
scores


max4rows = np.max(scores,axis=1)
max4rows


argmax4rows = np.argmax(scores,axis=1)
argmax4rows


scores[range(len(argmax4rows)),argmax4rows] = 0
scores


a = np.array([[1,2], [3, 4], [5, 6]])
a


a[a > 2] = 0
a


np.c_[a, np.ones(3).reshape(3,1)]


np.r_[a, np.ones(2).reshape(1,2)]


np.ones(2)


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[0,0,0]])
c = np.r_[a,b]
d = np.c_[a,b.T]
print(c,'\n',d)


scores


scores[:,0:-1]


scores[:,-1]


scores = np.random.random((5,3))-0.5
scores


(scores.T - np.max(scores, axis =1)).T


np.max(scores, axis =1)



