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
        self.max_iter = 8
        #tolerance
        self.eps_p = 1e-6
        self.eps_d = 1e-12
        self.eps_z = 1e-11 #zero

        self.lamda = None
        
        # Dual Formulation
        self.R = np.transpose(np.linalg.cholesky(self.H))
        self.Rinv = np.linalg.inv(self.R)
        # print(self.R)

        if not np.allclose(self.R, np.triu(self.R)):# check if upper triangular
            print("Not Upper!")

        self.M = None; self.v = None; self.d = None;

        #State Variables:
        self.lamda = None
        self.W = None
        self.W_hat = None

    def form_dual_objective(self):

        self.M = self.A.dot(self.Rinv)
        Rinv_T = self.Rinv.T
        self.v = Rinv_T.dot(self.f)
        self.d = self.b + self.M.dot(self.v)

        # print(self.M)
        # print(self.v)
        # print(self.d)

def solve(Prob):
    
    ## Working Set. Will keep increasing with iterations ##
    Prob.W = np.array([0])
    Prob.W_hat = np.setdiff1d(np.arange(Prob.m),Prob.W)
    Prob.lamda = np.zeros((Prob.m,1))
    Prob.lamda[0] = 3.0
    # print(Prob.W_hat)
    # print(Prob.W)
    # print(Prob.lamda)
    k = 0
    lamda_k = None
    lamda_k_W = None

    while k<Prob.max_iter:
        print(f'Iteration: {k+1}')
        M_k = Prob.M[Prob.W]
        # print("M_k: ", M_k)
        d_k = Prob.d[Prob.W]
        # print("d_k: ", d_k)

        M_k_hat = Prob.M[Prob.W_hat]
        d_k_hat = Prob.d[Prob.W_hat]
        mu_k = np.zeros((Prob.m, 1))
        ############ Process ###########
        '''
        MMT = (M_k)(M_k.T)
        '''
        MMT = M_k.dot(M_k.T) 
        if np.linalg.det(MMT) != 0:
            print(f'Step 3')
            #create sparse matrix
            MMT_sparse = sparse.csc_matrix(MMT)
            # if not MMT_sparse.sort_indices:
            #     MMT_sparse.sort_indices()

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
                    print("Converged!")
                    break
                else:
                    min_mu = np.inf
                    j = None
                    for i_w_hat in Prob.W_hat:
                        if mu_k[i_w_hat]<min_mu:
                            min_mu =  mu_k[i_w_hat]
                            j = i_w_hat
                
                    # j = Prob.W_hat[np.argmin(mu_k_W_hat)]
                    Prob.W = np.hstack((Prob.W, j)) #check output
                    Prob.W_hat = np.setdiff1d(np.arange(Prob.m),Prob.W)
                    print(f'Step 7: New W \n {Prob.W}')

            else:
                print('Step 9')
                p_k = lamda_k - Prob.lamda
                #index included in working set where lamda<0
                B = [] 
                for idx in Prob.W:
                    if lamda_k[idx][0]< Prob.eps_d:
                        B.append(idx)

                B = np.array(B)
                print("B: ", B)                

                print('Step 14')
                lamda_next, W_next = fix_component(Prob.lamda, Prob.W, B, p_k)
                print("lamda_next: ", lamda_next)
                Prob.lamda = lamda_next
                Prob.W = W_next

        else:
            print(f'Step 12:')
            U, s, V = np.linalg.svd(MMT)
            # extract the nullspace from the decomposition
            nullspace = V[np.argwhere(s < 1e-10).flatten()]
            print(nullspace)
            p_k = None

            for col in nullspace.T:
                # print(col)
                p_k = np.zeros(Prob.m)
                # print(Prob.W)
                p_k[Prob.W] = col
                if col.dot(p_k) < Prob.eps_z:
                    break
            
            print(f'Step 13')
            B = []
            for idx in Prob.W:
                if p_k[idx] < Prob.eps_z:
                    B.append(idx)
            B = np.array(B)
            print(f'B : {B}')

            Prob.lamda, Prob.W = fix_component(Prob.lamda, Prob.W, B, p_k)

        k = k+1

    print(f'Step 16')
    x = -1 * Prob.Rinv.dot(M_k.T.dot(lamda_k_W) + Prob.v)
    
    return x, Prob.lamda, Prob.W

            
def fix_component(lamda_k, W_k, B, p_k):
    
    print(f'Step 18')
    min_val = np.inf
    j = 0

    for i in B:
        temp = -1 * lamda_k[i][0]/p_k[i][0]
        if temp<min_val:
            min_val = temp
            j = i
            
    print(f'Old W_k: \n {W_k} \n j = {j}')
    #removes j from W_k
    W_new = np.setdiff1d(W_k,j) 
    print("New W_k", W_new)
    lamda_new = lamda_k - (lamda_k[j]/p_k[j]) * p_k

    return lamda_new, W_new



if __name__ == "__main__":

    H = np.array([[65.0, -22, -16],
              [-22.0, 14, 7],
              [-16, 7, 5]])

    f = np.expand_dims(np.array([3.0, 2.0, 3.0]),axis=1)
    # print(f)
    
    A = np.array([[1.0, 2.0, 1.0],
              [2.0, 0.0, 1.0],
              [-1.0, 2.0, -1.0]])

    b = np.expand_dims(np.array([3.0, 2.0, -2.0]), axis=1)

    # Instantiating Object
    prob = Active_set(H, f, A, b)
    
    # Forming Dual Objective
    prob.form_dual_objective()
    
    ans, lamda, W = solve(prob)
    print(f'Solution: \n {ans}')
