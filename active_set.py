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

    def iterate(self):
    

    def initialize(self):


        



def solve(Prob):
    
    Prob.W = np.zeros(self.H.shape[0])
    Prob.W[0] = 1

    

    while true:





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
    
    y = solve(prob)


    print(y)
