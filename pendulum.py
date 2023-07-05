import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# Define the system of ODEs
def lorenz(state,t):
    sigma = 10
    rho = 28
    beta = 8/3
    x, y, z = state  # Unpack the state vector
    return sigma*(y - x), x*(rho - z) - y, x*y - beta*z  # Derivatives

state0 = [0.0, 1.0, 1.0]  # Initial condition
t = np.arange(0.0, 50.0, 0.01)  # Time grid

# Solve the ODEs
state = odeint(lorenz, state0, t)

# Save the t and data in pkl format
data = np.concatenate((t.reshape(-1,1), state), axis=1)
df = pd.DataFrame(data, columns=['t','x','y','z'])
df.to_pickle('data/lorenz.pkl')

# Plot x against t
#plt.figure(figsize=(10,6))
#plt.plot(t, state[:, 1])
#plt.xlabel('t')
#plt.xlim(-5,25)
#plt.ylabel('x')
#plt.title('Lorenz Attractor - x vs t')
#plt.show()
