# Unser Problem:
# Gegeben ODE: x'= r*x*(1-x/K) = f(x) 
# min L(theta) = min sum (x_i - f(t_i))^2
# theta = (r,K,x0) Parametervektor
# x_i Messdaten und f(t_i) Modellvorhersage

# Analytisches Setting: f(t) = K/(1+((K-x0)/x0)*exp(-r*t))
# f(t) = K/(1+A*exp(-r*t)) mit A = (K-x0)/x0

MAX_Iter = int(1e5)
TOL_GRAD = 1e-7
TOL_L = 1e-9
MIN_Step_Size = 1e-5

data = np.loadtxt('testdaten.csv', delimiter=',', skiprows=1)
t_data = data[:,0]
x_data = data[:,1]
start_x0 = 2.0
start_K = 12.0
start_r = 1.0
theta = np.array([start_x0, start_r, start_K])

import numpy as np
# Numpy: Grundlegendes Paket für Mathematik in Python

def x(t, r, K, x0):
    A = (K-x0)/x0
    # Stabilisierung: vermeide Overflow in exp für große |r*t|
    exp_arg = np.clip(-r*t, -50.0, 50.0)
    return K/(1 + A*np.exp(exp_arg))

# f_x0 = - K/(1+A*exp(-r*t))^2 * dA/dx0 * exp(-r*t) 


# Compute Loss-Function L
def L(x0,r,K,t_data,x_data):
    simulate_x = x(t_data, r, K, x0)
    return np.sum((x_data - simulate_x)**2)

# Compute Gradient of L
# L' = -2 sum (x_i- x(t_1, theta)) * dx/dtheta
def gradL(x0,r,K,t_data,x_data):
    simulate_x = x(t_data, r, K, x0)
    residuals = x_data - simulate_x
    # Innere Ableitungen
    dL_dx0 = -2 * np.sum(residuals * dx_dx0(t_data, x0, K, r))
    dL_dK = -2 * np.sum(residuals * dx_dK(t_data, x0, K, r))
    dL_dr = -2 * np.sum(residuals * dx_dr(t_data, x0, K, r))
    return np.array([dL_dx0, dL_dr, dL_dK])

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
            alpha = MIN_Step_Size
            break
        alpha = alpha * beta 
        iteration = iteration + 1
    return alpha

for i in range(MAX_Iter):
    grad = gradL(*theta, t_data, x_data) #*y = Entpacken der Parameter

    #Schrittweiten Steuerung mit Armijo
    alpha = armijo_step(theta, grad, L, x, x_data, t_data)

    # Abbruch Bedingung:
    if np.linalg.norm(grad) < TOL_GRAD:
        break
    
    theta_new = theta - alpha * grad
    # clippe Werte um negative Parameter zu vermeiden
    theta_new = np.clip(theta_new, a_min=1e-6, a_max=None)

    # Weitere Abbruchbedingung: zu kleine Änderung in L 
    if np.abs(L(*theta_new, t_data, x_data) - L(*theta, t_data, x_data)) < TOL_L:
        break

    theta = theta_new

print(f"Geschätzte Parameter: x0={theta[0]}, r={theta[1]}, K={theta[2]}")
print(f"Minimale Verlustfunktion: {L(*theta, t_data, x_data)} \
      im Vergleich zu: {L(*np.array([1.0, 0.5, 10.0]), t_data, x_data)}")