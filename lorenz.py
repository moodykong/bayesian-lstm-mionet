import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from nn_lib import grf_1d, integrate, runge_kutta
import pickle

# define the Lorenz system
def lorenz(x, u):
    sigma = 10
    rho = 28
    beta = 8/3
    x, y, z = x
    f_x = sigma*(y - x)
    f_y = x*(rho - z) - y
    f_z = x*y - beta*z
    return np.array([f_x, f_y, f_z])

# for testing, we solve the following:
T = 40
h = 0.01
N = int(T/h)
x0 = np.array([0.,1.0,1.0])
n_spline = 120
x0 = np.hstack((np.random.uniform(-17,20,(n_spline,1)),np.random.uniform(-23,28,(n_spline,1)),np.random.uniform(0,50,(n_spline,1))))

def u_maker(func):
    def u(t, x):
        theta = x[0]
        theta_dot = x[1]
        return func(theta_dot)
    return u

def control(t, x):
    theta = x[0]
    theta_dot = x[1]
    return -0.80 * theta_dot
    #return np.sin(t/2)
    #return np.sin(t/2) - 0.80 * theta_dot
    

splines=[]

outputs={}
for i in range(n_spline):
    #spline = grf_1d()
    #splines.append(spline)
    #u = u_maker(spline)
    u = control
    soln = integrate(runge_kutta, lorenz, u, x0[i], h, N)
    x = np.vstack((x,soln.x[:-1,0].T)) if i>0 else (soln.x[:-1,0].T).reshape(1,-1)
    y = np.vstack((y,soln.x[:-1,1].T)) if i>0 else (soln.x[:-1,1].T).reshape(1,-1)
    z = np.vstack((z,soln.x[:-1,2].T)) if i>0 else (soln.x[:-1,2].T).reshape(1,-1)
    if i%100==0:
        print(f'{i} data are generated.')

outputs['x'] = x
outputs['y'] = y
outputs['z'] = z
outputs['t'] = soln.t

# save the data in pickle format
#filename = 'data/pendulum_random_init_2.pkl'
#filename = 'data/pendulum_test_random_init.pkl'
#filename = 'data/pendulum_test_random_init_stat.pkl'
#filename = 'data/lorenz_random_init.pkl'
filename = 'data/lorenz_random_test_init_t40_stat.pkl'

with open(filename, 'wb') as f:
    pickle.dump(outputs, f)
print(f'Data saved in {filename} .')
