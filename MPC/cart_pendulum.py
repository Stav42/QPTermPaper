import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import traj_calc
from qpsolvers import available_solvers, print_matrix_vector, solve_qp
# import active_set_prox
import active_set
import admmsolver
from scipy import sparse
import scipy.sparse.linalg as la
matplotlib.rcParams.update({'font.size': 22})

h = 10
Xt = traj_calc.get_traj(n_t=100, dt = 0.5)
duration = 100 * 0.5
dur = []
itr = []


def solve_asm(P, q, lb, ub):

    A = np.zeros((2 * P.shape[0], P.shape[1]))
    b = np.zeros((2 * P.shape[0], 1))
    A[:P.shape[0], :] = np.eye(P.shape[0])
    A[P.shape[0]:, :] = -1 * np.eye(P.shape[0])
    b[np.arange(P.shape[0])] = ub
    b[np.arange(start = P.shape[0], stop = 2*P.shape[0])] = -1*lb
    q = np.expand_dims(q, axis=1)
    # print(P.shape, q.shape, A.shape, b.shape)
    t1 = time.time()
    Prob = active_set.Active_set(H = P, f = q, A = A, b = b)
    Prob.form_dual_objective()
    iterr, x, lamda, W = active_set.solve(Prob)
    t2 = time.time()
    itr.append(iterr)
    dur.append(t2-t1)
    return x

def solve_admm(P, q, lb, ub):

    print(lb.shape)
    lb = np.array(lb[:, 0])
    print(lb.shape)
    ub = np.array(ub[:, 0])

    A = np.eye(P.shape[0])
    P = sparse.csc_matrix(P)
    A = sparse.csc_matrix(A)
    
    obj = admmsolver.ADMM(P, q, A, lb, ub)
    sol_x = None
    max_iter = 200
    curr_itr = 0 
    t1 = time.time()
    for i in range(0,max_iter):
        curr_itr += 1
        obj.solve()
        r_prim,r_dual,e_prim,e_dual = obj.residuals()
        
        if r_prim < e_prim and r_dual < e_dual:
            #unscale solution
            sol_x = obj.D.dot(obj.xk) 
            break

        #estimate new rho_o
        if i%200 == 0:
            old_rho_o = obj.rho_o
            obj.estimate_new_rho()
    #TODO : Unscale P, q as well
    opt_val = .5 * np.dot(sol_x, obj.P.dot(sol_x)) + \
        np.dot(obj.q, sol_x)
    opt_val = opt_val/obj.c

    print(f'Optimal objective : {opt_val}')

    print("Done")
    return sol_x

def solve_qpswift(P, c, ub, lb):

    G = np.zeros((2 * P.shape[0], P.shape[1]))
    h = np.zeros(2 * P.shape[0])
    G[:P.shape[0], :] = np.eye(P.shape[0])
    G[P.shape[0]:, :] = -1 * np.eye(P.shape[0])
    h[np.arange(P.shape[0])] = ub[:, 0]
    h[np.arange(start = P.shape[0], stop = 2*P.shape[0])] = -1*lb[:, 0]
    res = qpSWIFT.run(P = P, c = c, G = G, h = h)
    
    x = res['sol']
    return x



class CartPendulum:

    def __init__(self, state, M, m, b, l):
        self.state = state
        self.F =  None
        self.M = M
        self.m = m
        self.l = l
        self.b = b
        self.I = m*l**2/12 

    def MPC(self, Xt, t, hzn = 15, dt = 0.5):
        
        A = 0; B=0;
        for k in range(1, hzn+1):
            A += (dt * k)**2
            B += 2 * dt * k * (self.state -  Xt[:, t + k])

        # print("A and B: ", A, B)
        lb = -10 * np.ones([4, 1])
        ub =  10 * np.ones([4, 1])
            
        self.Xd = solve_asm(P = np.eye(4) * A, q = B, lb = lb, ub = ub)
        # self.Xd = solve_admm(P = np.eye(4) * A, q = B, lb = lb, ub = ub)
        # self.Xd = solve_qp(P = np.eye(4) * A, q = B, lb=lb, ub=ub, solver="qpswift") 
        # self.Xd = solve_qpswift(P = np.eye(4) * A, c = B, lb = lb, ub = ub)
        # self.Xd = qpSWIFT.run(P = np.eye(4) * A, c = B, lb = lb, ub = ub, solver = "qp_swift")


    def propagate_acc(self):

        dt = 0.5
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd; I = self.I;
       
        state[0] += self.Xd[0] * dt + self.Xd[1] * 0.5 * dt ** 2
        state[1] += self.Xd[1] * dt
        
        state[2] += self.Xd[2] * dt + self.Xd[3] * 0.5 * dt ** 2
        state[3] += self.Xd[3] * dt

        self.state = state

    def calculate_F(self):
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd;
        F = (M + m) * Xd[1] + b * Xd[0] + m * l * Xd[3]**2 * np.sin(state[2])
        return F

    def forward_dynamics(self, F, dt):
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd; I = self.I;
        A = np.array([[(M + m), m * l * np.cos(state[2])], [m * l * np.cos(state[2]), (I + m*l**2)]])
        B = np.array([[F + m * l * state[3] ** 2 * np.sin(state[2]) - b * state[1]], [-1*m*9.81*l*np.sin(state[2])]])

        X_dd = np.linalg.solve(A, B)

        state[1] += X_dd[0]*dt 
        state[0] += state[1] * dt + X_dd[0] * 0.5 * dt ** 2
        state[3] += X_dd[1]*dt 
        state[2] += state[3] * dt + X_dd[1] * 0.5 * dt ** 2

        self.state = state

def mse_error(a, b):

    a_len = a.shape[0]
    b_len = b.shape[0]

    length = min(a_len, b_len)

    s = 0
    for i in np.arange(length):
        s += (a[i] - b[i])**2

    c = s/length
    c = c**0.5
    return c
    
if __name__ == "__main__":

    x0 = np.array([0, 0, np.pi, 0]).T
    Pendulum = CartPendulum(x0, 2, 0.1, 1, 3)
    states = np.expand_dims(x0, axis=0)
    for i in range(85):
        Pendulum.MPC(Xt = Xt, t = i, dt = 0.5)
        # F = Pendulum.calculate_F()
        # Pendulum.forward_dynamics(F, dt = 0.5)
        Pendulum.propagate_acc()
        state_list = np.expand_dims(Pendulum.state, axis=0)
        states = np.vstack((states, state_list))

    Xt = Xt.T
    a = states[:, 0]
    b = Xt[:, 0]

    A = mse_error(a, b)

    c = states[:, 1]
    d = Xt[:, 1]
    e = states[:, 2]
    f = Xt[:, 2]

    B = mse_error(e, f)

    g = states[:, 3]
    h = Xt[:, 3]

    # print("MSE Over X: ", A)
    # print("MSE Over Theta: ", B)
    # # print(a.shape)
    # print(b.shape)
    plt.figure(0)
    plt.plot(a)
    plt.plot(b)

    plt.figure(1)
    plt.plot(c)
    plt.plot(d)

    plt.figure(2)
    plt.plot(e)
    plt.plot(f)

    plt.figure(3)
    plt.plot(g)
    plt.plot(h)

    plt.show()


        


        
         
    
