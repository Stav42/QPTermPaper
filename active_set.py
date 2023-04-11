import numpy as np



class Active_set:

    def __init__(self, H, f, A, b, lamda):

        # Primal Formulation
        self.H = H
        self.f = f
        self.A = A
        self.b = b

        self.lamda = None
        
        # Dual Formulation
        self.R = np.transpose(np.linalg.cholesky(self.H))

        if(!np.allclose(self.R, np.triu(self.R))):# check if upper triangular
            print(self.R)
            print("Not Upper!")

        self.M = None; self.v = None; self.d = None;

        #State Variables:
        self.lamda = None
        self.W = None
        self.W_hat = None

    def form_dual_objective(self):

        self.M = self.A * np.inverse(self.R)
        self.v = self.tranpose(self.inverse(self.R)) * self.f
        self.d = self.b + self.M * self.v



def solve(Prob):
    
    ## Working Set. Will keep increasing with iterations ##
    Prob.W = np.array([1])
    Prob.W_hat = np.arange(2, self.W.shape[0])

    Prob.lamda = np.zeros(self.W.shape[0])
    Prob.lamda[0] = 3

    k = 0;

    while true:
        
        M_k = Prob.M[Prob.W, :]
        d_k = Prob.d[Prob.W]

        M_k_hat = Prob.M[Prob.W_hat, :]
        d_k_hat = Prob.d[Prob.W_hat]

        ############ Process ###########

        if np.linalg.det(np.multiply(M_k, M_k.T)) != 0:
            
            factor = cholesky(sparse.csr_matrix(M_k * M_k.T))
            lamda_k = factor(-d_k)

            if np.all(lamda_k > 0):

                mu_k = M_k_hat * (M_k.transpose * lamda_k) + d_k_hat
                self.lamda = lamda_k

                if np.all(mu_k > 0):
                    break
                else:
                    j = np.argmin(mu_k)
                    Prob.W = np.hstack(Prob.W, Prob.W_hat[j])

            else:
                
                p_k = lamda_k - self.lamda
                indexes = np.where(lamda_k < 0)
                B_W = W_k[indexes]

                lamda_next, W_next = fix_component(Prob.lamda, Prob.W, B_w, p_k)
                self.lamda = lamda_next
                self.W = W_next

        else:

            U, s, V = np.linalg.svd(M_k * M_k.T)
            # extract the nullspace from the decomposition
            nullspace = V[np.argwhere(s < 1e-10).flatten()]

            p_k = None

            for col in nullspace.T:
                p_k = np.zeros(self.lamda_k.shape[0])
                p_k[Prob.W] = col
                if np.dot(col, p_k) < 0:
                    p_k_w = vol
                    break
            
            B = np.where(p_k < 0)
            B_w = Prob.W[B]

            Prob.lamda, Prob.W = fix_component(Prob.lamda, Prob.W, B_w, p_k)

        k = k+1

    x = -1 * np.inverse(self.R) * (M_k.T * lamda_k + Prob.v)
    
    return x, lamda_k, Prob.W

            
def fix_component(lamda_k, W_k, B, p_k):

    lamda_b = lamda_k[B]
    p_b = p_k[B]

    j = np.argmin( -1 * np.divide(lamda_b, p_b))
    W_k = W_k[W_k != j]
    lamda_k = lamda_k - lamda_k[j]/p_k[j] * p_k

    return lamda_k, W_k



if __name__ == "__main__":

    H = np.array([[65.0, -22, -16],
              [-22.0, 14, 7],
              [-16, 7, 5]])

    f = np.array([3.0, 2.0, 3.0])
    
    A = np.array([[1.0, 2.0, 1.0],
              [2.0, 0.0, 1.0],
              [-1.0, 2.0, -1.0]])

    b = np.array([3.0, 2.0, -2.0])

    # Instantiating Object
    prob = Active_set(H, f, A, b)
    
    # Forming Dual Objective
    prob.form_dual_objective()
    
    ans, lamda, W = solve(prob)
    print(ans)
