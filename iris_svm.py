import osqp
import qpSWIFT
from qpsolvers import solve_qp
# from qpsolvers import solve_qp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#prepare dataset ###################################
data = pd.read_csv('IRIS.csv')
print(data.head())
print(data['species'].unique())

data_new = data.loc[data['species'].isin(['Iris-setosa','Iris-virginica'])]
X = data_new.iloc[:,:4].values
y = data_new.iloc[:,4].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_enc,test_size=0.2,random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train[y_train==0] = -1
y_test[y_test==0] = -1

#OSQP problem #####################################
n = 4
m = 80
gamma = 1.0

b = y_train
Ad = sparse.csc_matrix(X_train)

Im = sparse.eye(m)

'''
min 1/2 xTx + 0Tx + 1/2tT0t + gamma*1Tt
x,t
s.t. diag(b)Ax + 1 <= t
     t>= 0
'''
P = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((m, m))], format='csc')
q = np.hstack([np.zeros(n), gamma*np.ones(m)])

'''
l>= ax >= u format:

-1 >= diag(b)Ax - t >= -inf
inf >= 0Tx + t >= 0

x = [x t] where
x -> weight vector
t -> dual
'''

A = sparse.vstack([
        sparse.hstack([sparse.diags(b).dot(Ad), -Im]),
        sparse.hstack([sparse.csc_matrix((m, n)), Im])
    ], format='csc')

l = np.hstack([-np.inf*np.ones(m), np.zeros(m)])
u = np.hstack([-np.ones(m), np.inf*np.ones(m)])

#solver
prob = osqp.OSQP()
prob.setup(P, q, A, l, u)
res = prob.solve()

#test model
wt = res.x[:4]
y_pred = X_test.dot(wt)
y_pred[y_pred<-1] = 1
y_pred[y_pred>1] = -1
y_pred = y_pred.astype('int')

#test accuracy
acc = accuracy_score(y_test,y_pred)

# QPSWIFT ############################################
opts = {'MAXITER':199,'VERBOSE':1,'OUTPUT':2}

n = 4
m = 80
gamma = 1.0

P = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((m, m))], format='csc').toarray()
c = np.hstack([np.zeros(n), gamma*np.ones(m)])

'''
diag(b)Ax - t <= -1
0Tx - t <= 0

x = [x t] where
x -> weight vector
t -> dual
'''

G = sparse.vstack([
        sparse.hstack([sparse.diags(b).dot(Ad), -Im]),
        sparse.hstack([sparse.csc_matrix((m, n)), -Im])
    ], format='csc').toarray()

h = np.hstack([-np.ones(m), np.zeros(m)])

#QP with inequality constraints
reseq = qpSWIFT.run(c,h,P,G,opts=opts)

#test model
wt = reseq['sol'][:4]
y_pred = X_test.dot(wt)
y_pred[y_pred<-1] = 1
y_pred[y_pred>1] = -1
y_pred = y_pred.astype('int')

#test accuracy
acc = accuracy_score(y_test,y_pred)

#QPOASES #############################################
n = 4
m = 80
gamma = 1.0

P = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((m, m))], format='csc').toarray()
q = np.hstack([np.zeros(n), gamma*np.ones(m)])

'''
diag(b)Ax - t <= -1
0Tx - t <= 0

x = [x t] where
x -> weight vector
t -> dual
'''

G = sparse.vstack([
        sparse.hstack([sparse.diags(b).dot(Ad), -Im]),
        sparse.hstack([sparse.csc_matrix((m, n)), -Im])
    ], format='csc').toarray()

h = np.hstack([-np.ones(m), np.zeros(m)])

A = np.zeros((1,n+m))
b = np.zeros(1)

# x = solve_qp(P,q,G,h,A,b,solver="qpoases")

# print(x)

