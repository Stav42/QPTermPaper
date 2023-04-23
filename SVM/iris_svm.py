import active_set_prox as asm
import admmsolver as admm
import qpSWIFT
from sksparse.cholmod import cholesky
import time
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(f'################# Dataset ##########################')
data = pd.read_csv('IRIS.csv')

data_new = data.loc[data['species'].isin(['Iris-setosa','Iris-virginica'])]
print(data_new.head())
print(data_new['species'].unique())

X = data_new.iloc[:,:4].values
y = data_new.iloc[:,4].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_enc,test_size=0.2,random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train[y_train==0] = -1
y_test[y_test==0] = -1

print(f'################### Solver : ADMM #########################')
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

max_iter = 600

if not P.has_sorted_indices:
    P.sort_indices()
if not A.has_sorted_indices:
    A.sort_indices()

obj = admm.ADMM(P,q,A,l,u) 
sol_x = None
tot_time = 0

print(f'Initial state : {obj.xk[:4]}')

for i in range(0,max_iter):
    # print(f'Iteration {i+1}')

    t1 = time.time()
    obj.solve()
    t2 = time.time()
    tot_time += (t2-t1)

    #termination status
    r_prim,r_dual,e_prim,e_dual = obj.residuals()

    if r_prim < e_prim and r_dual < e_dual:
        #unscale solution
        sol_x = obj.D.dot(obj.xk)[:4]
        print("Converged!")
        print(f'Iteration {i+1} and total time = {tot_time}s')
        break

    #estimate new rho_o
    if i%200 == 0:
        old_rho_o = obj.rho_o
        obj.estimate_new_rho()
        # if obj.rho_o != old_rho_o:
        #     print(f'Rho value changed to {obj.rho_o}')

#test model
wt = sol_x
y_pred = X_test.dot(wt)
y_pred[y_pred<-1] = 1
y_pred[y_pred>1] = -1
y_pred = y_pred.astype('int')

#test accuracy
acc = accuracy_score(y_test,y_pred)

print(f'Optimal weights : {wt}')
print(f'Test accuracy = {acc}')

print(f'################### Solver : Active-Set Method #########################')

n = 4
m = 80
gamma = 1.0

bd = y_train
Ad = sparse.csc_matrix(X_train)

Im = sparse.eye(m)

'''
min 1/2 xTx + 0Tx + 1/2tT0t + gamma*1Tt
x,t
s.t. diag(b)Ax + 1 <= t
     t>= 0
'''
H = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((m, m))], format='csc').toarray()
f = np.hstack([np.zeros(n), gamma*np.ones(m)])

'''
l>= ax >= u format:

diag(b)Ax - t <= -1
0Tx + -t <= 0 

x = [x t] where
x -> weight vector
t -> dual
'''

A = sparse.vstack([
        sparse.hstack([sparse.diags(bd).dot(Ad), -Im]),
        sparse.hstack([sparse.csc_matrix((m, n)), -Im])
    ], format='csc').toarray()

b = np.hstack([-np.ones(m), np.zeros(m)])

#solver

prob = asm.Active_set(H,f,A,b) 

prob.form_prox_dual_objective()

ans, lamda, W = asm.solve_prox(prob)

#test model
wt = ans[:4]
y_pred = X_test.dot(wt)
y_pred[y_pred<-1] = 1
y_pred[y_pred>1] = -1
y_pred = y_pred.astype('int')

#test accuracy
acc = accuracy_score(y_test,y_pred)

print(f'Optimal weights : {wt}')
print(f'Test accuracy = {acc}')

print(f'################### Solver : Interior Point Method #########################')
opts = {'MAXITER':199,'VERBOSE':1,'OUTPUT':2}

n = 4
m = 80
gamma = 1.0

b = y_train
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
print(f'Optimal weights : {wt}')
acc = accuracy_score(y_test,y_pred)
print(f'Test accuracy = {acc}')

