import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky



class Active_set:

    def __init__(self, H, f, A, b):

        # Primal Formulation
        self.H = H
        self.f = f
        self.A = A
        self.b = b

        self.lamda = None
        
        # Dual Formulation
        self.R = np.transpose(np.linalg.cholesky(self.H))

        if not np.allclose(self.R, np.triu(self.R)):# check if upper triangular
            print(self.R)
            print("Not Upper!")

        self.M = None; self.v = None; self.d = None;

        #State Variables:
        self.lamda = None
        self.W = None
        self.W_hat = None

    def form_dual_objective(self):

        self.M = np.matmul(self.A, np.linalg.inv(self.R))
        print("M zzz: ", self.M.shape)
        self.v = np.matmul(np.transpose(np.linalg.inv(self.R)), self.f)
        self.d = self.b + np.matmul(self.M, self.v)



def solve(Prob):
    
    ## Working Set. Will keep increasing with iterations ##
    Prob.W = np.array([0])
    Prob.W_hat = np.arange(1, Prob.M.shape[0])
    print(Prob.W_hat.shape)
    print(Prob.W)
    Prob.lamda = np.expand_dims(np.zeros(Prob.M.shape[0]), axis=1)
    Prob.lamda[0] = 3
    print(Prob.lamda.shape)
    print("Shapes")
    k = 0;

    while True:
        # print(Prob.d) 
        M_k = Prob.M[Prob.W, :]
        # print("M_k: ", M_k)
        d_k = Prob.d[Prob.W]
        print("d_k: ", d_k)

        M_k_hat = Prob.M[Prob.W_hat, :]
        d_k_hat = Prob.d[Prob.W_hat]
        # print("d_k_hat: ", d_k_hat.shape)

        ############ Process ###########
        
        # print("M_k * M_K.T: ", np.matmul(M_k, M_k.T))
        if np.linalg.det(np.matmul(M_k, M_k.T)) != 0:
            lamda_k = Prob.lamda.copy() 
            factor = cholesky(sparse.csr_matrix(np.matmul(M_k, M_k.T)))
            lamda_k_W = factor(-d_k)
            # lamda_k = factor(-d_k)

            print("W is: ", Prob.W)
            print("M_k is", M_k)
            lamda_k[Prob.W] = lamda_k_W
            print(lamda_k)
            ############# ISSUES HERE #########################
            if np.all(lamda_k > 0):
                print(M_k)
                print(lamda_k)
                mu_k = np.matmul(M_k_hat, np.matmul(M_k.transpose, lamda_k)) + d_k_hat
                self.lamda = lamda_k

                if np.all(mu_k > 0):
                    break
                else:
                    j = np.argmin(mu_k)
                    Prob.W = np.hstack(Prob.W, Prob.W_hat[j])

            else:
                print("chalu: ")
                p_k = lamda_k - Prob.lamda
                p_k[0] = 3;
                indexes = []
                for idx, row in enumerate(lamda_k):
                    if row[0]<0 and idx in Prob.W:
                        indexes.append(idx)

                indexes = np.array(indexes)
                print("indexes: ", indexes)
                B_w = Prob.W[indexes]
                

                lamda_next, W_next = fix_component(Prob.lamda, Prob.W, B_w, p_k)
                print("lamda_next: ", lamda_next)
                Prob.lamda = lamda_next.copy()
                Prob.W = W_next.copy()

        else:
            # print(np.linalg.det(np.matmul(M_k, M_k.T)))
            U, s, V = np.linalg.svd(np.matmul(M_k,  M_k.T))
            # print(M_k.shape)
            # print(np.matmul(M_k, M_k.T).shape)
            # extract the nullspace from the decomposition
            nullspace = V[np.argwhere(s < 1e-10).flatten()]
            # print(nullspace)
            p_k = None

            for col in nullspace.T:
                # print(col)
                p_k = np.zeros(Prob.lamda.shape[0])
                # print(Prob.W)
                p_k[Prob.W] = col
                if np.dot(col, p_k) < 0:
                    p_k_w = vol
                    break
            
            B = np.where(p_k < 0)
            B_w = Prob.W[B]

            Prob.lamda, Prob.W = fix_component(Prob.lamda, Prob.W, B_w, p_k)

        k = k+1

    x = -1 * np.matmul(np.inverse(self.R), (np.matmul(M_k.T, lamda_k) + Prob.v))
    
    return x, lamda_k, Prob.W

            
def fix_component(lamda_k, W_k, B, p_k):
    
    # print("fix Component")
    print(lamda_k)
    print(B)
    lamda_b = lamda_k[B]
    # p_b = p_k[B]

    min_val = np.inf
    j = 0

    for i in B:
        temp = -1 * np.divide(lamda_k[i], p_k[i])
        if temp<min_val:
            min_val = temp
            j = i
            
    print("Purana: ", W_k, j)
    W_k = W_k[W_k != j]
    print("Purana2: ", W_k, j)
    lamda_k = lamda_k - (lamda_k[j]/p_k[j]) * p_k

    return lamda_k, W_k



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
    print(ans)
