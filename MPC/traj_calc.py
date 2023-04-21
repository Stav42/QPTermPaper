import numpy as np
import matplotlib.pyplot as plt

def return_poly(d, c, b, a, t):
    x = a*t**3 + b*t**2 + c*t + d;
    x_d = 3*a*t**2 + 2*b*t + c;
    
    return [x, x_d]

## Cubic Fitting 
def spline_fit(x0, x0d, x1, x1d, n_t, duration):
     
    M = np.array([[1, 0, 0, 0], 
                 [0, 1, 0, 0], 
                 [1, duration, duration**2, duration**3], 
                 [0, 1, 2*duration, 3*duration**2]])

    [d, c, b, a] = np.linalg.solve(M, np.array([x0, x0d, x1, x1d]))
    t_steps = np.linspace(start = 0, stop = duration, num = n_t, endpoint = True)
    traj = []
    for t in t_steps:
      traj.append(return_poly(d, c, b, a, t))      
    
    return np.array(traj)


## Trajectory : R[4x100]
def get_traj(n_t = 100, dt = 0.5, phase_ratio = [0.4, 0.4, 0.2]):
    
    n_t_1 = int(phase_ratio[0] * n_t)
    n_t_2 = int(phase_ratio[1] * n_t)
    n_t_3 = int(phase_ratio[2] * n_t)

    dur_1 = n_t_1 * dt
    dur_2 = n_t_2 * dt
    dur_3 = n_t_3 * dt

    T_1 = np.zeros([4, n_t_1])
    T_2 = np.zeros([4, n_t_2])
    T_3 = np.zeros([4, n_t_3])

    T_1[:, 0] = np.array([0, 0, np.pi, 0])
    T_2[:, 0] = np.array([3, 2, np.pi/2, 0])
    T_3[:, 0] = np.array([2, 0, 0, -np.pi/6])
    T_3[:, -1] = np.array([0, 0, np.pi, 0])
    
    ### FOR FIRST PHASE ###############
    
    X_1 = T_1[:, 0]
    X_2 = T_2[:, 0]
    tx = spline_fit(X_1[0], X_1[1], X_2[0], X_2[1], n_t_1, dur_1)
    tt = spline_fit(X_1[2], X_1[3], X_2[2], X_2[3], n_t_1, dur_1)
    t = np.hstack((tx, tt))
    t1 = np.transpose(t)

    ##### FOR SECOND PHASE ####################
    
    X_2 = T_2[:, 0]
    X_3 = T_3[:, 0]
    tx = spline_fit(X_2[0], X_2[1], X_3[0], X_3[1], n_t_2, dur_2)
    tt = spline_fit(X_2[2], X_2[3], X_3[2], X_3[3], n_t_2, dur_2)
    t = np.hstack((tx, tt))
    t2 = np.transpose(t)

    #### FOR THIRD PHASE ####################

    X_3 = T_3[:, 0]
    X_4 = T_3[:, -1]
    tx = spline_fit(X_3[0], X_3[1], X_4[0], X_4[1], n_t_3, dur_3)
    tt = spline_fit(X_3[2], X_3[3], X_4[2], X_4[3], n_t_3, dur_3)
    t = np.hstack((tx, tt))
    t3 = np.transpose(t)
 
    t_traj = np.hstack((t1, t2))
    t_traj = np.hstack((t_traj, t3))

    # print(t_traj.shape)
    # print(t_traj)

    # plt.plot(t_traj[0, :])
    # plt.show()


# get_traj()

