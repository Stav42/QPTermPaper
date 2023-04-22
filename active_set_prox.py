import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky

class Active_set:

    def __init__(self, H, f, A, b):

        # Primal Formulation
        self.m = A.shape[0] #number of constraints
        self.H = H
        self.f = f
        self.A = A
        self.b = b
        self.max_iter = 20
        self.max_prox_iter = 20
        #proximal
        self.e = 1e-4
        self.eta = 1.5e-8
        #tolerance
        self.eps_p = 1e-6
        self.eps_d = 1e-12
        self.eps_z = 1e-11 #zero

        self.lamda = None
        self.R = None       

        self.M = None; self.v = None; self.d = None;

        #State Variables:
        self.lamda = None
        self.W = None
        self.W_hat = None

    def form_prox_dual_objective(self):
        #Proximal point
        H = self.H + self.e*np.eye(self.H.shape[0])
        self.R = np.transpose(np.linalg.cholesky(H))

        if not np.allclose(self.R, np.triu(self.R)):# check if upper triangular
            print("Not Upper!")

        self.Rinv = np.linalg.inv(self.R)
        self.M = self.A.dot(self.Rinv)
        #v & d computed later 

def fix_prox_component(lamda_k, W_k, B, p_k):
    
    print(f'Step 18')
    min_val = np.inf
    j = 0

    for i in B:
        temp = -1 * lamda_k[i]/p_k[i]
        if temp<min_val:
            min_val = temp
            j = i
            
    print(f'Old W_k: \n {W_k} \n j = {j}')
    #removes j from W_k
    W_new = np.setdiff1d(W_k,j) 
    print("New W_k", W_new)
    lamda_new = lamda_k - (lamda_k[j]/p_k[j]) * p_k

    return lamda_new, W_new

def solve_prox(Prob):
    
    Prob.W = np.array([0])
    Prob.W_hat = np.setdiff1d(np.arange(Prob.m),Prob.W)
    Prob.lamda = np.zeros(Prob.m)
    Prob.lamda[0] = 3.0
    #optimal sol
    x = np.zeros(Prob.H.shape[0])
    i = 0
    lamda_k = None
    lamda_k_W = None

    while i<Prob.max_prox_iter:
        i = i+1
        Rinv_T = Prob.Rinv.T
        Prob.v = Rinv_T.dot(Prob.f - Prob.e*x)
        Prob.d = Prob.b + Prob.M.dot(Prob.v)
        # print(f'Shape : {Prob.d.shape}')

        x_old = x

        #inner loop
        k = 0       

        while k<Prob.max_iter:
            print(f'Iteration: {k+1}')
            M_k = Prob.M[Prob.W]
            d_k = Prob.d[Prob.W]

            M_k_hat = Prob.M[Prob.W_hat]
            d_k_hat = Prob.d[Prob.W_hat]
            mu_k = np.zeros(Prob.m)
            
            ############ Process ###########
            '''
            MMT = (M_k)(M_k.T)
            '''
            MMT = M_k.dot(M_k.T) 
            if np.linalg.det(MMT) != 0:
                print(f'Step 3')
                #create sparse matrix
                MMT_sparse = sparse.csc_matrix(MMT)

                lamda_k = Prob.lamda.copy() #copy by value
                factor = cholesky(MMT_sparse)
                lamda_k_W = factor(-d_k)

                print("W : \n ", Prob.W)
                print("M_k : \n ", M_k)
                lamda_k[Prob.W] = lamda_k_W
                print(f'lambda_k : \n {lamda_k}')

                if np.all(lamda_k >= -Prob.eps_d):
                    print(f'Step 5')
                    mu_k_W_hat = M_k_hat.dot(M_k.T.dot(lamda_k_W)) + d_k_hat
                    mu_k[Prob.W_hat] = mu_k_W_hat
                    Prob.lamda = lamda_k

                    if np.all(mu_k > -Prob.eps_p):
                        print("Inner loop convereged!")
                        break
                    else:
                        min_mu = np.inf
                        j = None
                        for i_w_hat in Prob.W_hat:
                            if mu_k[i_w_hat]<min_mu:
                                min_mu =  mu_k[i_w_hat]
                                j = i_w_hat
                    
                        Prob.W = np.hstack((Prob.W, j)) 
                        Prob.W_hat = np.setdiff1d(np.arange(Prob.m),Prob.W)
                        print(f'Step 7: New W \n {Prob.W}')

                else:
                    print('Step 9')
                    p_k = lamda_k - Prob.lamda
                    #index included in working set where lamda<0
                    B = [] 
                    for idx in Prob.W:
                        if lamda_k[idx]< Prob.eps_d:
                            B.append(idx)

                    B = np.array(B)
                    print("B: ", B)                

                    print('Step 14')
                    lamda_next, W_next = fix_prox_component(Prob.lamda, Prob.W, B, p_k)
                    print("lamda_next: ", lamda_next)
                    Prob.lamda = lamda_next
                    Prob.W = W_next
                    Prob.W_hat = np.setdiff1d(np.arange(Prob.m),Prob.W)

            else:
                print(f'Step 12:')
                U, s, V = np.linalg.svd(MMT)
                # extract the nullspace from the decomposition
                nullspace = V[np.argwhere(s < 1e-10).flatten()]
                print(nullspace)
                p_k = np.zeros(Prob.m)

                for col in nullspace.T:
                    p_k[Prob.W] = col.copy()
                    if col.dot(Prob.d) < Prob.eps_z:
                        break
                
                print(f'Step 13')
                B = []
                for idx in Prob.W:
                    if p_k[idx] < Prob.eps_z:
                        B.append(idx)
                B = np.array(B)
                print(f'B : {B}')

                lamda_next, W_next = fix_prox_component(Prob.lamda, Prob.W, B, p_k)
                Prob.lamda = lamda_next.copy()
                Prob.W = W_next.copy()
                Prob.W_hat = np.setdiff1d(np.arange(Prob.m),Prob.W)

            k = k+1

        print(f'Step 16')
        x = -1 * Prob.Rinv.dot(M_k.T.dot(lamda_k_W) + Prob.v)

        if np.linalg.norm(x-x_old,ord=2) < Prob.eta:
            print(f'Converged!')
            break

    return x, Prob.lamda, Prob.W