import numpy as np
import matplotlib.pyplot as plt
import traj_calc
from qpsolvers import available_solvers, print_matrix_vector, solve_qp

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

    def MPC(self, Xt, hzn = 8, dt = 0.5):
        
        for i in range(len(Xt)):

            A = 0; B=0;
            for k in range(hzn):
                A += (dt * k)**2
                B += -2 * dt * k * Xt[:, i+k]

            lb = -3 * np.ones([4, 1])
            ub =  3 * np.ones([4, 1])

            self.Xd = solve_qp(P = np.eye(4) * A, q = B, lb = lb, ub = ub, solver = "osqp")

    def calculate_F(self):
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd;
        F = (M + m) * Xd[1] + b * Xd[0] + m * l * Xd[3]**2 * np.sin(state[2])
        return F

    def forward_dynamics(self, F, dt):
        M = self.M; m = self.m; b = self.b; l = self.l; state = self.state; Xd = self.Xd; I = self.I;
        A = np.array([[(M + m), m * l * np.cos(state[2])], [m * l * np.cos(state[2]), (I + m*l**2)]])
        B = np.array([[F + m * l * state[3] ** 2 * np.sin(state[2]) - b * state[1]], [-1*m*9.81*l*np.sin(state[2])]])

        X_dd = np.linalg.solve(A, B)

        state[1] += X_dd[0] * dt
        state[0] += state[1] * dt + 0.5 * X_dd[0] * dt**2
        
        state[3] += X_dd[1] * dt
        state[2] += state[3] * dt + 0.5 * X_dd[1] * dt**2

        self.state = state
        # print(state)

    
if __name__ == "__main__":

    x0 = np.array([0, 2, np.pi, 0]).T
    Pendulum = CartPendulum(x0, 5, 3, 3, 2)
    states = []
    for i in range(100):
        Pendulum.MPC(Xt = Xt, dt = 0.5)
        F = Pendulum.calculate_F()
        Pendulum.forward_dynamics(F, dt = 0.5)
        states.append(Pendulum.state)

    # print(states)
    # print(len(states))
    # a = np.array(states)
    # print(a)
    # a = a[:, 0]
    # plt.plot(a)
    # plt.show()

        


        
         
    
