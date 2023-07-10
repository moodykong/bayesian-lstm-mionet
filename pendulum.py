import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from nn_lib import grf_1d, integrate, runge_kutta
import pickle

# define the pendulu's vector field
m = 1
l = 1
I = (1 / 12) * m * (l ** 2)
b = 0.01
g = 9.81
def pendulum(x, u):
    theta, omega = x
    f_theta = omega
    f_omega = (1 / (0.25 * m * (l ** 2) + I)) * (u - b * omega - 0.5 * m * l * g * np.sin(theta))
    return np.array([f_theta, f_omega])

# for testing, we solve the following:
T = 10
h = 0.01
N = int(T/h)
x = np.array([0.1,1.0])
n_spline = 120
x = np.random.rand(n_spline,2)
x[:,0] = x[:,0] * 2 * np.pi - np.pi
x[:,1] = x[:,1] * 2 *8 - 8
x[:,1] *= 0.

def u_maker(func):
    def u(t, x):
        theta, theta_dot = x
        return func(theta_dot)
    return u

def control(t, x):
    theta, theta_dot = x
    return -0.80 * theta_dot
    #return np.sin(t/2)
    #return np.sin(t/2) - 0.80 * theta_dot
    

splines=[]

outputs={}
for i in range(n_spline):
    spline = grf_1d()
    splines.append(spline)
    #u = u_maker(spline)
    u = control
    soln = integrate(runge_kutta, pendulum, u, x[i], h, N)
    theta = np.vstack((theta,soln.x[:-1,0].T)) if i>0 else (soln.x[:-1,0].T).reshape(1,-1)
    theta_dot = np.vstack((theta_dot,soln.x[:-1,1].T)) if i>0 else (soln.x[:-1,1].T).reshape(1,-1)
    if i%100==0:
        print(f'{i} data are generated.')

outputs['theta'] = theta
outputs['omega'] = theta_dot
outputs['t'] = soln.t

# save the data in pickle format
#filename = 'data/pendulum_random_init_2.pkl'
#filename = 'data/pendulum_test_random_init.pkl'
filename = 'data/pendulum_test_random_init_stat.pkl'
with open(filename, 'wb') as f:
    pickle.dump(outputs, f)
print(f'Data saved in {filename} .')
