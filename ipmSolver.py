import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky

# ne - number of equality constraint
# ni - number of inequality constraint

class QP:

    def __init__(self, P, G, A, c, h, b):
        
        self.P = P; self.G = G; self.A = A
        self.c = c; self.h = h; self.b = b

        self.P_n = P.shape[0]; self.A_n = A.shape[0]; self.G_n = G.shape[0]

        self.ni = G.shape[0]; self.ne = A.shape[0]

        self.P_csc = None
        self.G_csc = None
        self.A_csc = None

        # K_b is stacked -c, b, h | K_w if matrix in eqn 20
        self.K = None; self.K0 = None; self.K_b = None; self.K_w = None

        # Current state
        self.xk = None; self.yk = None; self.zk = None; self.sk = None; self.lamda = None

    def CCS(self):
        self.P_csc = sparse.csc_matrix(P)
        self.G_csc = sparse.csc_matrix(G)
        self.A_csc = sparse.csc_matrix(A)

    def initialize(self):
        
        P_n = self.P.shape[0]
        A_n, A_m = self.A.shape
        G_n, G_m = self.G.shape

        self.K = np.zeros([P_n + A_n + G_n, P_n + A_n + G_n])
        self.K[0:P_n, 0:P_n] = self.P
        self.K[P_n:P_n+A_n, 0:A_m] = self.A
        self.K[P_n + A_n:, 0:G_m] = self.G

        self.K[0:A_m, P_n:P_n+A_n] = self.A.T
        self.K[0:G_m, P_n + A_n:] = self.G.T
        self.K[-G_n:, -G_n:] = -np.eye(G_n)

        self.K0 = self.K

        K_b = np.hstack((-self.c, self.b))
        K_b = np.hstack((K_b, self.h)).T
        self.K_b = np.expand_dims(K_b,axis=1) #shape: 6x1

        factor = cholesky(sparse.csr_matrix(self.K))
        Dir = factor(K_b) #shape: (6,)
        
        x_0 = np.expand_dims(Dir[:P_n],axis=1)
        y_0 = np.expand_dims(Dir[P_n:P_n+A_n],axis=1)
        
        z_cap = Dir[-G_n:]

        a_p = np.amax(z_cap)
        if a_p<0:
            s_0 = -z_cap
        else:
            s_0 = -z_cap + (1+a_p)*np.ones(G_n)
        
        a_d = np.amax(-z_cap)
        if a_d<0:
            z_0 = z_cap
        else:
            z_0 = z_cap + (1+a_d)*np.ones(G_n)

        z_0 = np.expand_dims(z_0,axis=1)
        s_0 = np.expand_dims(s_0,axis=1)

        return x_0, y_0, z_0, s_0
    
    
    def affine_direction(self):

        self.K0[-self.G_n:, -self.G_n:] = np.zeros([self.G_n, self.G_n])
        K_b = -self.K_b #shape : 6x1

        state = np.vstack((self.xk, self.yk))
        state = np.vstack((state, self.zk)) #shape : 6x1

        s_state = np.zeros(state.shape)
        s_state[-self.sk.shape[0]:,:] = self.sk
        # print(f's_state : {s_state},{s_state.shape}')

        r = s_state + np.dot(self.K0, state) + K_b

        self.lamda = np.sqrt(np.multiply(self.sk, self.zk))

        #LHS of eqn 20
        self.K_w = self.K0
        Wt_W = np.multiply(np.eye(self.G_n),np.divide(self.sk,self.zk))
        self.K_w[-self.G_n:,-self.G_n:] = -Wt_W

        #RHS of eqn 20
        r_b = -r
        r_b[-self.sk.shape[0]:,:] += self.sk

        #solve eqn 20
        factor = cholesky(sparse.csr_matrix(self.K_w))
        del_state = factor(r_b)

        return r, del_state, Wt_W
    
    def compute_centering_params(self, del_state_a, Wt_W):
        
        # eqn 19
        del_za = del_state_a[-self.G_n:]
        del_sa = -np.dot(Wt_W, del_za) - self.sk

        alpha_p = 1e10
        alpha_d = 1e10
        flag1 = 0
        flag2 = 0

        for i in range(self.sk.shape[0]):
            if del_sa[i]<0 and (-self.sk[i]/del_sa[i])<alpha_p:
                alpha_p = -self.sk[i]/del_sa[i]
                flag1 = 1
        
            if del_za[i]<0 and (-self.zk[i]/del_za[i])<alpha_d:
                alpha_d = -self.zk[i]/del_za[i]
                flag2 = 1
        
        if(flag1==0):
            alpha_p=1
        if(flag2==0):
            alpha_d=1
        
        rho_n = np.dot((self.sk + alpha_p*del_sa).T,(self.zk + alpha_d*del_za))
        rho_d = np.dot(self.sk.T,self.zk)
        rho = rho_n/rho_d

        sigma = max(0,(min(1,rho))**3)
        
        return del_sa, sigma, 1

    def correction_step(self, del_sa, sigma, del_state_a, r, Wt_W):
        # Mehrotra Steps
        G_n = self.G.shape[0]
        del_za = del_state_a[-G_n:]
        mu = np.multiply(self.sk, self.zk)
        rs = -np.multiply(self.lamda, self.lamda) - np.multiply(del_sa, del_za) + np.multiply(sigma, mu)

        r_b = -r
        r_b[-self.sk.shape[0]:] += np.divide(rs, self.zk)
        
        factor = cholesky(sparse.csr_matrix(self.K_w))
        cent_del_state = factor(r_b)
        print(f'cent_del_state : {cent_del_state}')
        del_s = -np.dot(Wt_W, cent_del_state[-self.G_n:]) - np.divide(rs, self.zk)
        print(f'del_s : {del_s}')
        return del_s, cent_del_state
    
    def solve(self):
              
        self.xk, self.yk, self.zk, self.sk = self.initialize()

        for i in range(10):
            
            r, del_state, Wt_W = self.affine_direction()

            # del_s = -np.dot(Wt_W, del_state[-self.G_n:]) - self.sk
            del_sa, sigma, alpha = self.compute_centering_params(del_state, Wt_W)
            del_s, cent_del_state = self.correction_step(del_sa, sigma, del_state, r, Wt_W)
            # print(alpha, cent_del_state)
            self.xk +=  alpha*cent_del_state[:self.P_n]
            self.yk +=  alpha*cent_del_state[self.P_n: self.P_n + self.A_n]
            self.zk +=  alpha*cent_del_state[-self.G_n:]
            self.sk +=  alpha*del_s

            print("Iteration No: ", i)

            # print(del_state)

            nrm = max(del_state)
            if nrm<10e-10:
                print("converged")
                break
            print("xk, yk, zk, sk: ", self.xk, self.yk, self.zk, self.sk)

        print("Done")
        # print(sigma)




# 1. Initialization of x0, y0, z0, s0,
# 2. Problem Setup (input of variables)
# 3. (Search Direction) Predictor Corrector Step



if __name__ == "__main__":

    P = np.array([[5.0,1.0,0.0],
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 4.0]])

    c = np.array([1.0,2.0,1.0])

    ### Inequality Constraints
    G = np.array([[-4.0,-4.0,0.0],
        [0.0,0.0,-1.0]])

    h = np.array([-1.0,-1.0])

    ### Equality Constraints
    A = np.array([[1.0, -2.0, 1.0]])

    b = np.array([3.0])

    QPEx = QP(P, G, A, c, h, b)
    QPEx.CCS()

    QPEx.solve()
    
