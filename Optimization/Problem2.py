# Unser Problem:
# Gegeben ODE: x'= r*x*(1-x/K) = f(x) 
# min L(theta) = min sum (x_i - f(t_i))^2
# theta = (r,K,x0) Parametervektor
# x_i Messdaten und f(t_i) Modellvorhersage

# Analytisches Setting: f(t) = K/(1+((K-x0)/x0)*exp(-r*t))
# f(t) = K/(1+A*exp(-r*t)) mit A = (K-x0)/x0

import numpy as np
# Numpy: Grundlegendes Paket für Mathematik in Python


# f_x0 = - K/(1+A*exp(-r*t))^2 * dA/dx0 * exp(-r*t) 
def df_dx0(t,x0,K,r):
    A = (K - x0) / x0
    return K**2 * np.exp(-r*t) / (x0**2) * (1+A * np.exp(-r*t))**2

def df_dK(t,x0,K,r):
    A = (K - x0) / x0
    term1 = 1/(1 + A * np.exp(-r*t))
    term2 = K * np.exp(-r*t) / (x0 * (1 + A * np.exp(-r*t))**2)
    return term1 - term2

def df_dr(t,x0,K,r):
    A = (K - x0) / x0
    return K * A * t * np.exp(-r*t) / (1 + A * np.exp(-r*t))**2

# Compute Loss-Function L
def L(x0,r,K,t_data,x_data):
    evaluate_f = f(t_data, r, K, x0)
    return np.sum((x_data - evaluate_f)**2)


data = np.loadtxt('testdaten.csv', delimiter=',', skiprows=1)
t_data = data[:,0]
x_data = data[:,1]

MAX_Iter = 1000
TOL_GRAD = 1e-3
TOL_L = 1e-4
MIN_Step_Size = 1e-4

# Compute Sensitivities: S_x0' = f_x0, S_x0(0) = 1
# Erinnerung S' = df/dx * S + df/dtheta
# dx/dt  = f(x, theta) -> d/d theta dx/dt = d/d theta f = f_x * dx/d theta + f_theta

Num_Steps = len(t_data)
h = t_data[1] - t_data[0]

# f = r*x*(1-x/K) = rx - r/K x^2 -> f_x = r - 2r/K * x
def S_x0(t, r, K, x0, x_grid):
    S_x0 = np.zeros(Num_Steps)
    S_x0[0] = 1.0

    # Euler Verfahren 
    for i in range(1, Num_Steps):
        S_x0[i] = S_x0[i-1] + (-2*r/K * x_grid[i-1] +r) * S_x0[i-1] * h
    return S_x0

def S_r(t, r, K, x0, x_grid):
    S_r = np.zeros(Num_Steps)
    S_r[0] = 0.0

    # Euler Verfahren 
    for i in range(1, Num_Steps):
        S_r[i] = S_r[i-1] + (-2*r/K * x_grid[i-1] +r) * S_r[i-1] * h + x_grid[i-1] * (1 - x_grid[i-1]/K) * h
    return S_r

def S_K(t, r, K, x0, x_grid):
    S_K = np.zeros(Num_Steps)
    S_K[0] = 0.0

    # Euler Verfahren 
    for i in range(1, Num_Steps):
        S_K[i] = S_K[i-1] + (-2*r/K * x_grid[i-1] +r) * S_K[i-1] * h + r/K**2 * x_grid[i-1]**2 * h
    return S_K

def x_simulated(t, r, K, x0):
    x_simulated = np.zeros(Num_Steps)
    x_simulated[0] = x0 

    # Euler Verfahren
    # x' = r*x*(1-x/K) -> x_{n+1} = x_n + r*x_n*(1-x_n/K)*h
    for i in range(1, Num_Steps):
        x_simulated[i] = x_simulated[i-1] + r * x_simulated[i-1] * (1 - x_simulated[i-1]/K) * h
    return x_simulated

# Compute Gradient of L
# L' = -2 sum (x_i- f(t_1, theta)) * df/dtheta
def gradL(x0,r,K,t_data,x_data):
    # Berechne Trajektorie x aus den gegebenen Parametern
    x_grid = x_simulated(t_data, r, K, x0)
    grad_x0 = 2*(x_grid - x_data) * S_x0(t_data, r, K, x0, x_grid)
    grad_r = 2*(x_grid - x_data) * S_r(t_data, r, K, x0, x_grid)
    grad_K = 2*(x_grid - x_data) * S_K(t_data, r, K, x0, x_grid)
    return np.array([np.sum(grad_x0), np.sum(grad_r), np.sum(grad_K)])

start_x0 = 1.5
start_K = 12.0
start_r = 1.0

theta = np.array([start_x0, start_r, start_K])

# Armijo Bedingung 
# L(theta_new) <= L(theta) - c * alpha * ||gradL(theta)||^2
def armijo_step(theta,grad, L,f, x_data, t_data, alpha_init =1.0,c=1e-4, beta=0.5):
    Forced_Exit = False
    alpha = alpha_init
    iteration = 0
    L_current = L(*theta, t_data, x_data)

    # Jetzt schrittweitensuche
    while True:
        theta_new = theta - alpha * grad
        L_new = L(*theta_new, t_data, x_data)
        if L_new < L_current - c * alpha * np.linalg.norm(grad)**2:
            break
        if iteration > 1000:
            Forced_Exit = True
            break
        alpha = alpha * beta 
        iteration = iteration + 1
    
    if Forced_Exit == True:
        alpha = MIN_Step_Size
    return alpha

for i in range(MAX_Iter):
    grad = gradL(*theta, t_data, x_data) #*y = Entpacken der Parameter

    #Schrittweiten Steuerung mit Armijo
    alpha = armijo_step(theta, grad, L, f, x_data, t_data)

    # Abbruch Bedingung:
    if np.linalg.norm(grad) < TOL_GRAD:
        break
    
    theta_new = theta - alpha * grad

    # Weitere Abbruchbedingung: zu kleine Änderung in L 
    if np.abs(L(*theta_new, t_data, x_data) - L(*theta, t_data, x_data)) < TOL_L:
        break

    theta = theta_new

print(f"Geschätzte Parameter: x0={theta[0]}, r={theta[1]}, K={theta[2]}")