import numpy as np
from scipy import sparse
# from scipy.sparse.linalg.dsolve import linsolve
import qdldl

class ADMM():
    
    def __init__(self,P,q,A,l,u):
        
        self.A = A; self.P = P; self.q = q; self.l = l; self.u = u
        #cold start
        self.xk = np.zeros(q.shape); self.yk = np.zeros((A.shape[0],1)); self.zk = np.zeros((A.shape[0],1))
    
        #admm params
        self.sigma = 1.6
        self.alpha = 1e-6
        
        #vectorized rho
        rho_o = 0.1
        self.rho = np.zeros((A.shape[0],1))
        self.rho_inv = np.zeros((A.shape[0],1))
        for i in range(A.shape[0]):
            if (u[i,0]-l[i,0])<0.01:
                self.rho[i,0] = (10**3)*rho_o
            else:
                self.rho[i,0] = rho_o
            
            self.rho_inv[i,0] = 1./self.rho[i,0] 

        #K matrix in step 1
        p0,p1 = P.shape; a0,a1 = A.shape
        self.K = np.zeros((p0+a0, p1+a0))
        self.K[:p0,:p1] = P + self.sigma*np.eye(p0)
        self.K[p0:,:a1] = A
        self.K[:a1,p0:] = A.T
        
        val = np.squeeze(self.rho_inv)
        m1 = np.eye(p1+a0-a1)
        np.fill_diagonal(m1,val)
        self.K[a1:,a1:] = -m1
        print(self.K)

        #LDL factorization
        K = sparse.csr_matrix(self.K)
        self.F = qdldl.Solver(K)
    
    def linearsolver(self,rhs):

        sol = self.F.solve(rhs)
        return sol
    
    def create_rhs(self):
        rhs = np.zeros((self.P.shape[0]+self.A.shape[0],1))
        rhs[:self.q.shape[0],:] = self.sigma*self.xk - self.q
        rhs[self.q.shape[0]:,:] = self.zk - np.multiply(self.rho_inv,self.yk)
        return rhs

    def projection(self,x):
        ans = np.zeros(x.shape)
        for i in range(x.shape[0]):
            if x[i,:] <= self.l[i,:]:
                ans[i,:] = self.l[i,:]
            elif x[i,:] >= self.u[i,:]:
                ans[i,:] = self.u[i,:]
            else:
                ans[i,:] = x[i,:]
        return ans
    
    def solve(self):
        #Part 3 of algorithm 1: uses xk,yk,zk to solve
        rhs = self.create_rhs()    
        xz_tilde =  self.linearsolver(rhs)
        xtilde = np.expand_dims(xz_tilde[:self.P.shape[0]],axis=1)
        nu = np.expand_dims(xz_tilde[self.P.shape[0]:],axis=1)

        #update variables
        ztilde = self.zk + np.multiply(self.rho_inv , (nu-self.yk))
        self.xk = self.alpha*xtilde + (1-self.alpha)*self.xk
        num1 = self.alpha*ztilde + (1-self.alpha)*self.zk + np.multiply(self.rho_inv,self.yk)
        z_prev = self.zk 
        self.zk = self.projection(num1)        
        num2 = self.alpha*ztilde + (1-self.alpha)*z_prev - self.zk
        self.yk = self.yk + np.multiply(self.rho,num2)
    
    def residuals(self):
        v1 = np.dot(self.A,self.xk)
        v2 = np.dot(self.P,self.xk)
        v3 = np.dot(self.A.T,self.yk)
        r_prim = np.amax(np.abs(v1 - self.zk))
        r_dual = np.amax(np.abs(v2 + self.q + v3))

        e_prim = 0.001 + 0.001*max(np.amax(np.abs(v1)),np.amax(np.abs(self.zk)))
        e_dual = 0.001 + 0.001*max(np.amax(np.abs(v2)),np.amax(np.abs(v3)),np.amax(np.abs(self.q)))

        return r_prim,r_dual,e_prim,e_dual

if __name__ == '__main__':

    #https://github.com/osqp/osqp/blob/master/tests/basic_qp/generate_problem.py

    P = np.array([[5.0,1.0,0.0],
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 4.0]])

    q = np.array([[1.0],
                [2.0],
                [1.0]])

    A = np.array([[1.0, -2.0, 1.0],
                [-4.0,-4.0,0.0],
                [0.0,0.0,-1.0]])
    
    l = np.array([[3.0],
                [-np.inf],
                [-np.inf]])
    
    u = np.array([[3.0],
                [-1.0],
                [-1.0]])

    #tolerance
    max_iter = 100
    i=0

    #warm start
    obj = ADMM(P,q,A,l,u)
    obj.solve()
    r_prim,r_dual,e_prim,e_dual = obj.residuals()
    print(r_prim,r_dual,e_prim,e_dual)
    
    
    print(f'Initial state : {obj.xk}')

    while r_prim>e_prim and r_dual>e_dual and i<max_iter:
        i = i+1
        print(f'Iteration : {i}')
        print(f'Residual : primal = {r_prim} > {e_prim} , dual = {r_dual} > {e_dual}')
        obj.solve()
        r_prim,r_dual,e_prim,e_dual = obj.residuals()
        print(f'Updated state : {obj.xk}')