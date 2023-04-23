#https://github.com/osqp/osqp-python/blob/master/src/osqppurepy/_osqp.py

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as la
class ADMM():
    
    def __init__(self,P,q,A,l,u):
        
        self.n = P.shape[0]
        self.m = A.shape[0]
        self.q = q 
        self.P = sparse.csc_matrix((P.data,P.indices,P.indptr),shape=(self.n,self.n))
        self.A = sparse.csc_matrix((A.data,A.indices,A.indptr),shape=(self.m,self.n))
        self.rho = np.zeros(self.m)
        self.rho_o = 0.1
        self.rho_rng = [1e-6,1e+6]
        self.adap_rho_tol = 5
        self.infty = 1e+30
        self.scale = [1e-4,1e+4] #[min_scaling, max_scaling]
        self.first_run = 1

        self.l = np.maximum(l,-self.infty)
        self.u =  np.minimum(u,self.infty)

        # print(self.P.toarray())
        # print(self.A.toarray())
        # print(self.q)
        # print(self.l)
        # print(self.u)

        #cold start
        self.xk = np.zeros(self.n); self.yk = np.zeros(self.m); self.zk = np.zeros(self.m)

        #store previous values
        self.x_prev = None; self.z_prev = None
        
        #admm params
        self.sigma = 1e-6
        self.alpha = 1.6
        self._scaling = 10

        #scale params
        self.D = None
        self.E = None
        self.Dinv = None
        self.Einv = None
        self.c = None
        
        #setup 
        if self.first_run:
            self.scaler()
            self.setup_rho()
            self.setup_K_matrix()
            self.first_run = 0
        
    def setup_rho(self):
        #vectorized rho
        rho_o = np.minimum(np.maximum(self.rho_o,self.rho_rng[0]),self.rho_rng[1])

        loose = np.where(np.logical_and(self.l<-self.infty*self.scale[0],self.u>self.infty*self.scale[1]))[0]
        eq = np.where((self.u-self.l)<1e-4)[0]
        ineq = np.setdiff1d(np.setdiff1d(np.arange(self.m),eq),loose)
        self.rho[loose] = self.rho_rng[0]
        self.rho[eq] = (1e3)*rho_o
        self.rho[ineq] = rho_o
        self.rho_inv = np.reciprocal(self.rho)
        #call setup_K_matrix() everytime after setup_rho() is called

    def setup_K_matrix(self):
        #K matrix in step 1
        K1 = self.P + self.sigma*sparse.eye(self.n)
        K2 = -sparse.diags(self.rho_inv)
        kkt = sparse.vstack([
            sparse.hstack([K1,self.A.T]),
            sparse.hstack([self.A,K2])
        ])

        #factorization
        self.K = la.splu(kkt.tocsc())
    
    def linearsolver(self,rhs):
        sol = self.K.solve(rhs)
        return sol

    def K_norm_cols(self,P,A):
        P_norm = la.norm(P,np.inf,axis=0)
        A_norm = la.norm(A,np.inf,axis=0)
        norm_ist = np.maximum(P_norm,A_norm)
        norm_2nd = la.norm(A,np.inf,axis=1)

        return np.hstack((norm_ist,norm_2nd))

    def limit(self,vec):
        try:
            for i in range(len(vec)):
                if vec[i]<self.scale[0]:
                    vec[i] = 1.0
                elif vec[i]>self.scale[1]:
                    vec[i] = self.scale[1]
        except:
            if vec<self.scale[0]:
                vec = 1.0
            elif vec>self.scale[1]:
                vec = self.scale[1]
            
        return vec

    def scaler(self):
        s = np.ones(self.n+self.m)
        c = 1.0

        P = self.P
        q = self.q
        A = self.A
        l = self.l
        u = self.u

        '''
        S = [[D    ],
             [    E]]
        '''
        D = sparse.eye(self.n)
        E = sparse.eye(self.m)

        for i in range(self._scaling):
            inf_norm_cols = self.K_norm_cols(P,A)
            inf_norm_cols = self.limit(inf_norm_cols)
            s = np.reciprocal(np.sqrt(inf_norm_cols))

            D_temp = sparse.diags(s[:self.n])
            E_temp = sparse.diags(s[self.n:])

            P = D_temp.dot(P.dot(D_temp)).tocsc()
            A = E_temp.dot(A.dot(D_temp)).tocsc()
            q = D_temp.dot(q)
            l = E_temp.dot(l)
            u = E_temp.dot(u)

            D = D_temp.dot(D)
            E = E_temp.dot(E)

            #cost
            inf_norm_P = la.norm(P,np.inf,axis=0).mean()
            inf_norm_q = self.limit(np.linalg.norm(q, np.inf))
            cost = self.limit(np.maximum(inf_norm_P,inf_norm_q))
            gamma = 1./cost

            P = gamma*P
            q = gamma*q
            c = gamma*c

            self.P = sparse.csc_matrix((P.data,P.indices,P.indptr),shape=(self.n,self.n))
            self.A = sparse.csc_matrix((A.data,A.indices,A.indptr),shape=(self.m,self.n))
            self.q = q
            self.l = l
            self.u = u

            self.D = D
            self.Dinv = sparse.diags(np.reciprocal(D.diagonal()))
            self.E = E
            self.Einv = sparse.diags(np.reciprocal(E.diagonal()))
            self.c = c
        
        return
    
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
        self.x_prev = np.copy(self.xk)
        self.z_prev = np.copy(self.zk)
        rhs = self.create_rhs()   

        xz_tilde =  self.linearsolver(rhs)
        
        ##
        # print(f'xz_tilde {xz_tilde}')

        #ztilde
        ztilde = xz_tilde[self.n:]
        xz_tilde[self.n:] = self.z_prev + self.rho_inv*(ztilde - self.yk)

        #update variables
        self.xk = self.alpha*xz_tilde[:self.n] + (1-self.alpha)*self.x_prev
        
        num1 = self.alpha*xz_tilde[self.n:] + (1-self.alpha)*self.z_prev + self.rho_inv*self.yk
        self.zk = self.projection(num1)        
        
        num2 = self.alpha*xz_tilde[self.n:] + (1-self.alpha)*self.z_prev - self.zk
        self.yk = self.yk + self.rho*num2 
    
    def residuals(self):
        #Computed for scaled QP
        v1 = self.Einv.dot(self.A.dot(self.xk) - self.zk)
        r_prim = np.linalg.norm(v1,np.inf)

        v2 = self.P.dot(self.xk) + self.q + self.A.T.dot(self.yk)
        v2 = (1./self.c)*self.Dinv.dot(v2)
        r_dual = np.linalg.norm(v2,np.inf)

        max_prim = np.max([
                        np.linalg.norm(self.Einv.dot(self.A.dot(self.xk)),np.inf),
                        np.linalg.norm(self.Einv.dot(self.zk),np.inf)
        ])
        max_dual = (1./self.c)*np.max([
                        np.linalg.norm(self.Dinv.dot(self.A.T.dot(self.yk)),np.inf),
                        np.linalg.norm(self.Dinv.dot(self.P.dot(self.xk)),np.inf),
                        np.linalg.norm(self.Dinv.dot(self.q),np.inf)
        ])

        e_prim = 0.001 + 0.001*max_prim
        e_dual = 0.001 + 0.001*max_dual

        return r_prim,r_dual,e_prim,e_dual

    def estimate_new_rho(self):
        P = self.P
        A = self.A
        q = self.q

        # Compute normalized residuals
        r_prim = np.linalg.norm(A.dot(self.xk) - self.zk, np.inf)
        r_prim /= (np.max([np.linalg.norm(A.dot(self.xk), np.inf),
                            np.linalg.norm(self.zk, np.inf)]) + 1e-10)
        r_dual = np.linalg.norm(P.dot(self.xk) + q + A.T.dot(self.yk), np.inf)
        r_dual /= (np.max([np.linalg.norm(A.T.dot(self.yk), np.inf),
                           np.linalg.norm(P.dot(self.xk), np.inf),
                           np.linalg.norm(q, np.inf)]) + 1e-10)

        # Compute new rho
        rho_o_new = self.rho_o * np.sqrt(r_prim/(r_dual + 1e-10))
        rho_o_new = np.minimum(np.maximum(rho_o_new, self.rho_rng[0]), self.rho_rng[1])

        if rho_o_new>=0 and \
            ((rho_o_new > self.adap_rho_tol*self.rho_o) or (rho_o_new < (1./self.adap_rho_tol)*self.rho_o)):
             self.rho_o = rho_o_new
             #update rho and rho_inv vectors
             self.setup_rho()
             #update KKT matrix
             self.setup_K_matrix()

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

    max_iter = 50

    #setup P,A
    P = sparse.csc_matrix(P)
    A = sparse.csc_matrix(A)

    if not P.has_sorted_indices:
        P.sort_indices()
    if not A.has_sorted_indices:
        A.sort_indices()

    obj = ADMM(P,q,A,l,u) 
    sol_x = None
    
    print(f'Initial state : {obj.xk}')

    for i in range(0,max_iter):
        print(f'Iteration {i+1}')

        obj.solve()

        #termination status
        print(f'x = {obj.xk}, y = {obj.yk}, z = {obj.zk}')

        r_prim,r_dual,e_prim,e_dual = obj.residuals()
        
        print(f'Primal res = {r_prim}, Primal tol = {e_prim}')
        print(f'Dual res = {r_dual}, Dual tol = {e_dual}')

        if r_prim < e_prim and r_dual < e_dual:
            #unscale solution
            sol_x = obj.D.dot(obj.xk) 
            print("Converged!")

            print(f'Final x = {sol_x}')
            break

        #estimate new rho_o
        if i%200 == 0:
            old_rho_o = obj.rho_o
            obj.estimate_new_rho()
            if obj.rho_o != old_rho_o:
                print(f'Rho value changed to {obj.rho_o}')

    #TODO : Unscale P, q as well
    opt_val = .5 * np.dot(sol_x, obj.P.dot(sol_x)) + \
        np.dot(obj.q, sol_x)
    opt_val = opt_val/obj.c

    print(f'Optimal objective : {opt_val}')


    print("Done")
