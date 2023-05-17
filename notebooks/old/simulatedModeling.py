# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# %%
alpha = gamma = 1

# %%
def lotkaVolterra(p, t=0):
    """
    Inputs parameters (p) followed by independent variable
    """
    beta = 0.01
    delta = 0.02

    dpdt = ([p[0] * (1 - beta*p[1]), p[1] * (-1 + delta*p[0])])
    return dpdt

t = np.linspace(0, 15, 1000)
p0 = [50, 50]

# odeint takes the function, initial values, then defined indepdent variable
x, infodict = odeint(lotkaVolterra, p0, t, full_output=True)
print(infodict['message'])

# %%
plt.plot(t, x[:,0])
plt.plot(t, x[:,1])

# %% Resistant sensitive
S0 = 0.01
R0 = 0
Vd = 0.1
Vc = 0.9
eps = 1e-6
d = 1
pr = 0.2
uon = 1.5
ton = 1
toff = 3

alpha = 10**-2

def u(t):
    return 1.5

def sensResFunc(SR, t = 0):
    S = SR[0]
    R = SR[1]
    dS = (1-(S+R))*S - (eps+alpha*u(t))*S - d*u(t)*S
    dR = pr*(1-(S+R))*R + (eps + alpha*u(t))*S

    return [dS, dR]


t = np.linspace(0, 80, 10000)
x, infodict = odeint(sensResFunc, [S0, R0], t, full_output=True)
print(infodict['message'])

# %%
S = x[:,0]
R = x[:,1]
v = S+R
# plt.plot(t, x[:,0], label='Sensitive')
# plt.plot(t, x[:,1], label = 'Resistant')
plt.plot(t, v, label = 'Volume')
plt.legend()
plt.grid()
# %%
