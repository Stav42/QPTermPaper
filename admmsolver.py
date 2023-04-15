#https://github.com/osqp/osqp-python/blob/master/src/osqppurepy/_osqp.py

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as la
import qdldl

class ADMM():
    
    def __init__(self,P,q,A,l,u):
        
        self.n = P.shape[0]
        self.m = l.shape[0]
        self.q = q; self.l = l; self.u = u
        self.P = sparse.csc_matrix(P)
        self.A = sparse.csc_matrix(A)
        
        #cold start
        self.xk = np.zeros(self.n); self.yk = np.zeros(self.m); self.zk = np.zeros(self.m)

        #store previous values
        self.x_prev = None; self.z_prev = None
        #admm params
        self.sigma = 1e-6
        self.alpha = 1.6
        
        #vectorized rho
        rho_o = 0.1
        self.rho = np.zeros(self.m)
        eq = np.where((self.u-self.l)<1e-4)[0]
        ineq = np.setdiff1d(np.arange(self.m),eq)
        self.rho[eq] = (1e3)*rho_o
        self.rho[ineq] = rho_o
        self.rho_inv = np.reciprocal(self.rho)

        #K matrix in step 1
        K1 = self.P + self.sigma*sparse.eye(self.n)
        K2 = -sparse.diags(self.rho_inv)
        kkt = sparse.vstack([
            sparse.hstack([K1,self.A.T]),
            sparse.hstack([A,K2])
        ])

        #factorization
        self.K = la.splu(kkt.tocsc())
    
    def linearsolver(self,rhs):
        sol = self.K.solve(rhs)
        return sol
    
    def create_rhs(self):
        rhs = np.zeros(self.n+self.m)
        rhs[:self.n] = self.sigma*self.x_prev - self.q
        rhs[self.n:] = self.z_prev - self.rho_inv*self.yk
        return rhs

    def projection(self,x):
        ans = np.minimum(np.maximum(x,self.l),self.u)
        return ans
    
    def solve(self):
        #Part 3 of algorithm 1: uses xk,yk,zk to solve
        rhs = self.create_rhs()    
        xz_tilde =  self.linearsolver(rhs)
        #ztilde
        xz_tilde[self.n:] = self.z_prev + self.rho_inv*(xz_tilde[self.n:] - self.yk)

        #update variables
        self.xk = self.alpha*xz_tilde[:self.n] + (1-self.alpha)*self.x_prev
        
        num1 = self.alpha*xz_tilde[self.n:] + (1-self.alpha)*self.z_prev + self.rho_inv*self.yk
        self.zk = self.projection(num1)        
        
        num2 = self.alpha*xz_tilde[self.n:] + (1-self.alpha)*self.z_prev - self.zk
        self.yk = self.yk + self.rho*num2 
    
    def residuals(self):
        v1 = self.A.dot(self.xk) - self.zk
        r_prim = np.linalg.norm(v1,np.inf)

        v2 = self.P.dot(self.xk) + self.q + self.A.T.dot(self.yk)
        r_dual = np.linalg.norm(v2,np.inf)

        max_prim = np.max([
                        np.linalg.norm(self.A.dot(self.xk),np.inf),
                        np.linalg.norm(self.zk,np.inf)
        ])
        max_dual = np.max([
                        np.linalg.norm(self.A.T.dot(self.yk),np.inf),
                        np.linalg.norm(self.P.dot(self.xk),np.inf),
                        np.linalg.norm(self.q,np.inf)
        ])

        e_prim = 0.001 + 0.001*max_prim
        e_dual = 0.001 + 0.001*max_dual

        return r_prim,r_dual,e_prim,e_dual

if __name__ == '__main__':

    #https://github.com/osqp/osqp/blob/master/tests/basic_qp/generate_problem.py

    P = np.array([[5.0,1.0,0.0],
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 4.0]])

    q = np.array([1.0,2.0,1.0])

    A = np.array([[1.0, -2.0, 1.0],
                [-4.0,-4.0,0.0],
                [0.0,0.0,-1.0]])
    
    l = np.array([3.0,-np.inf,-np.inf])
    
    u = np.array([3.0,-1.0,-1.0])

    max_iter = 10

    obj = ADMM(P,q,A,l,u) 
    
    print(f'Initial state : {obj.xk}')

    for i in range(0,max_iter):
        print(f'Iteration {i}')
        obj.x_prev = np.copy(obj.xk)
        obj.z_prev = np.copy(obj.yk)

        obj.solve()

        print(f'Current state : {obj.xk}, {obj.yk}, {obj.zk}')
        r_prim,r_dual,e_prim,e_dual = obj.residuals()
        print(f'Residuals : {r_prim}, {r_dual}')
        print(f'Tolerance : {e_prim}, {e_dual}')
        if r_prim < e_prim and r_dual < e_dual:
            print("Converged!")
            break
    print("Done")