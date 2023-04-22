import numpy as np
import matplotlib.pyplot as plt
import traj_calc
from qpsolvers import available_solvers, print_matrix_vector, solve_qp

h = 10
Xt = traj_calc.get_traj(n_t=100, dt = 0.5)
duration = 100 * 0.5

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

        self.Xd = solve_qp(P = np.eye(4) * A, q = B, lb = lb, ub = ub, solver = "osqp")

        print(self.Xd)

    def propagate_acc(self):

        dt = 0.5
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd; I = self.I;
       
        print(self.Xd[0], self.Xd[1])
        state[0] += self.Xd[0] * dt + self.Xd[1] * 0.5 * dt ** 2
        print(state[0])
        # state[0] += state[1] * dt + 0.5 * self.Xd[0] * dt**2
        state[1] += self.Xd[1] * dt
        
        state[2] += self.Xd[2] * dt + self.Xd[3] * 0.5 * dt ** 2
        # state[2] += state[3] * dt + 0.5 * self.Xd[1] * dt**2
        state[3] += self.Xd[3] * dt

        self.state = state

    def calculate_F(self):
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd;
        F = (M + m) * Xd[1] + b * Xd[0] + m * l * Xd[3]**2 * np.sin(state[2])
        # print(F)
        return F

    def forward_dynamics(self, F, dt):
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd; I = self.I;
        A = np.array([[(M + m), m * l * np.cos(state[2])], [m * l * np.cos(state[2]), (I + m*l**2)]])
        B = np.array([[F + m * l * state[3] ** 2 * np.sin(state[2]) - b * state[1]], [-1*m*9.81*l*np.sin(state[2])]])

        X_dd = np.linalg.solve(A, B)

      #  print(self.Xd[0], self.Xd[1])
        state[1] += X_dd[0]*dt 
        state[0] += state[1] * dt + X_dd[0] * 0.5 * dt ** 2
       # print(state[0])
        # state[0] += state[1] * dt + 0.5 * self.Xd[0] * dt**2
        state[3] += X_dd[1]*dt 
        state[2] += state[3] * dt + X_dd[1] * 0.5 * dt ** 2

        self.state = state
        # print(state)

    
if __name__ == "__main__":

    x0 = np.array([0, 0, np.pi, 0]).T
    Pendulum = CartPendulum(x0, 2, 0.1, 1, 3)
    states = np.expand_dims(x0, axis=0)
    for i in range(85):
        # print("i : ", i)
        Pendulum.MPC(Xt = Xt, t = i, dt = 0.5)
        F = Pendulum.calculate_F()
        Pendulum.forward_dynamics(F, dt = 0.5)
 #       Pendulum.propagate_acc()
        state_list = np.expand_dims(Pendulum.state, axis=0)
        states = np.vstack((states, state_list))

    # print(states)
    print(states.shape)
    Xt = Xt.T
    a = states[:, 0]
    b = Xt[:, 0]
    c = states[:, 1]
    d = Xt[:, 1]
    e = states[:, 2]
    f = Xt[:, 2]
    g = states[:, 3]
    h = Xt[:, 3]

    # print(a.shape)
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


        


        
         
    
