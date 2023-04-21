import numpy as np
import traj_calc
import qp_solvers

Xt = traj_calc.get_traj(n_t=100, dt = 0.5)
duration = n_t * dt




def CartPendulum:

    def __init__(self, state, M, m, b, l):
        self.state = state
        self.F =  None
        self.M = M
        self.m = m
        self.l = l
        self.b = b
        self.I = m*l**2/12

    def MPC(self, Xt, hzn = 8, dt):
        
        for i in range(len(Xt)):

            A = 0; B=0;
            for k in range(hzn):
                A += (dt * k)**2
                B += -2 * dt * k * Xt[:, i+k]

            lb = -3 * np.ones([4, 1])
            ub =  3 * np.ones([4, 1])

            self.Xd = QPSolve(A, B, lb, ub)

    def calculate_F(self):
       
        F = (M + m) * Xd[1] + b * Xd[0] + m * l * Xd[3]**2 * np.sin(state[2])
        return F

    def forward_dynamics(F, dt):

        A = np.array([[(M + m), m * l * np.cos(state[2])], [m * l * np.cos(state[2]), (I + m*l**2)]])
        B = np.array([[F + m * l * state[3] ** 2 * np.sin(state[2]) - b * state[1]], [-1*m*9.81*l*np.sin(state[2])]])

        X_dd = np.linalg.solve(A, B)

        state[1] += X_dd[0] * dt
        state[0] += state[1] * dt + 0.5 * X_dd[0] * dt**2
        
        state[3] += X_dd[1] * dt
        state[2] += state[3] * dt + 0.5 * X_dd[1] * dt**2

        self.state = state

    
if __name__ == "__main__":

    x0 = np.array([0, 2, np.pi, 0]).T
    Pendulum = CartPendulum(x0, 5, 3, 3, 2)

        

        


        
         
    
