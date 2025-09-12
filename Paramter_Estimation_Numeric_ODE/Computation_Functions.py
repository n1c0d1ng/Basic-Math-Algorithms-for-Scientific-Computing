import numpy as np
from config import*

data = np.loadtxt("testdaten.csv", delimiter=",", skiprows=1)
anzahl_messdaten = data.shape[0]

faktor = 1

NUM_Steps = anzahl_messdaten * faktor
h = FINAL_TIME / (NUM_Steps - 1)

# Definition des dynamischen Systenms
# x' = f(x) = r * x * (1 - x/K)
def x(x0, K, r, return_full=False):
    x_grid = np.zeros(NUM_Steps)
    x_grid[0] = x0
    for i in range(1, NUM_Steps):
        x_grid[i] = x_grid[i-1] + r * x_grid[i-1] * (1 - x_grid[i-1]/K) * h
        x_grid[i] = np.clip(x_grid[i], 1e-8, 1e4)  # Beispielgrenzen
    if return_full == True:
        return x_grid
    else:
        return x_grid[np.arange(0, NUM_Steps, faktor)]

# Definition Sensitivity Equations
#------------------------------------------------------------------------------------------------
def Sensitivity_x0(x0, K, r, x_grid):
    s_x0 = np.zeros(NUM_Steps)
    s_x0[0] = 1.0

    # Euler Verfahren zur numerischen Lösung der Sensitivitätsgleichung
    # S' = f_x * S + f_x0 = f_x * S = (-2r/K * x + r) * S
    for i in range (1, NUM_Steps):
        s_x0[i] = s_x0[i-1] + (-2*r/K * x_grid[i-1] + r) * s_x0[i-1] * h
    return s_x0[np.arange(0, NUM_Steps, faktor)]

#------------------------------------------------------------------------------------------------
def Sensitivity_K(x0, K, r, x_grid):
    s_K = np.zeros(NUM_Steps)
    s_K[0] = 0.0

    # Euler Verfahren zur numerischen Lösung der Sensitivitätsgleichung
    # S' = f_x * S + f_K = (-2r/K * x + r) * S + r * x^2 / K^2
    for i in range (1, NUM_Steps):
        s_K[i] = s_K[i-1] + (-2*r/K * x_grid[i-1] + r) * s_K[i-1] * h \
        + (r * x_grid[i-1]**2 / K**2) * h
    return s_K[np.arange(0, NUM_Steps, faktor)]

#------------------------------------------------------------------------------------------------
def Sensitivity_r(x0, K, r, x_grid):
    s_r = np.zeros(NUM_Steps)
    s_r[0] = 0.0

    # Euler Verfahren zur numerischen Lösung der Sensitivitätsgleichung
    # S' = f_x * S + f_r = (-2r/K * x + r) * S + x - x^2/K
    for i in range (1, NUM_Steps):
        s_r[i] = s_r[i-1] + (-2*r/K * x_grid[i-1] + r) * s_r[i-1] * h \
        + (x_grid[i-1] - x_grid[i-1]**2 / K) * h
    return s_r[np.arange(0, NUM_Steps, faktor)]


# Loss Function
# L = sum_i=1^  (x(t_i) - x_data_i)^2 -> min
def L(x0, K, r, x_data):
    x_grid = x(x0, K, r, return_full=False)
    return np.sum((x_grid - x_data)**2)

# Gradient der Loss Function
def grad_L(x0, K, r, x_data):
    grad_x0 = 2 * (x(x0, K, r,return_full=False) - x_data) * Sensitivity_x0(x0, K, r, x(x0, K, r, return_full=True))
    grad_K = 2 * (x(x0, K, r,return_full=False) - x_data) * Sensitivity_K(x0, K, r, x(x0, K, r, return_full=True))
    grad_r = 2 * (x(x0, K, r,return_full=False) - x_data) * Sensitivity_r(x0, K, r, x(x0, K, r, return_full=True))
    return np.array([np.sum(grad_x0), np.sum(grad_K), np.sum(grad_r)])
