import numpy as np
from config import*

# Definition des dynamischen Systenms
# x' = f(x) = r * x * (1 - x/K)
def x(x0, K, r):
    time = np.linspace(0, FINAL_TIME, NUM_Steps)
    x_grid = np.zeros(len(time))
    x_grid[0] = x0
    for i in range(1, len(time)):
        x_grid[i] = x_grid[i-1] + r * x_grid[i-1] * (1 - x_grid[i-1]/K) * (FINAL_TIME/NUM_Steps)
        x_grid[i] = np.clip(x_grid[i], 1e-8, 1e4)  # Beispielgrenzen
    return x_grid

# Definition Sensitivity Equations
#------------------------------------------------------------------------------------------------
def Sensitivity_x0(x0, K, r, x_grid):
    time = np.linspace(0, FINAL_TIME, NUM_Steps)  # Zeitintervall
    s_x0 = np.zeros(len(time))
    s_x0[0] = 1.0

    # Euler Verfahren zur numerischen Lösung der Sensitivitätsgleichung
    # S' = f_x * S + f_x0 = f_x * S = (-2r/K * x + r) * S
    for i in range (1, len(time)):
        s_x0[i] = s_x0[i-1] + (-2*r/K * x_grid[i-1] + r) * s_x0[i-1] * (FINAL_TIME/NUM_Steps)
    return s_x0

#------------------------------------------------------------------------------------------------
def Sensitivity_K(x0, K, r, x_grid):
    time = np.linspace(0, FINAL_TIME, NUM_Steps)  # Zeitintervall
    s_K = np.zeros(len(time))
    s_K[0] = 0.0

    # Euler Verfahren zur numerischen Lösung der Sensitivitätsgleichung
    # S' = f_x * S + f_K = (-2r/K * x + r) * S + r * x^2 / K^2
    for i in range (1, len(time)):
        s_K[i] = s_K[i-1] + (-2*r/K * x_grid[i-1] + r) * s_K[i-1] * (FINAL_TIME/NUM_Steps) \
        + (r * x_grid[i-1]**2 / K**2) * (FINAL_TIME/NUM_Steps)
    return s_K

#------------------------------------------------------------------------------------------------
def Sensitivity_r(x0, K, r, x_grid):
    time = np.linspace(0, FINAL_TIME, NUM_Steps)  # Zeitintervall
    s_r = np.zeros(len(time))
    s_r[0] = 0.0

    # Euler Verfahren zur numerischen Lösung der Sensitivitätsgleichung
    # S' = f_x * S + f_r = (-2r/K * x + r) * S + x - x^2/K
    for i in range (1, len(time)):
        s_r[i] = s_r[i-1] + (-2*r/K * x_grid[i-1] + r) * s_r[i-1] * (FINAL_TIME/NUM_Steps) \
        + (x_grid[i-1] - x_grid[i-1]**2 / K) * (FINAL_TIME/NUM_Steps)
    return s_r


# Loss Function
# L = sum_i=1^  (x(t_i) - x_data_i)^2 -> min
def L(x0, K, r, x_data):
    x_grid = x(x0, K, r)
    return np.sum((x_grid - x_data)**2)

# Gradient der Loss Function
def grad_L(x0, K, r, x_data):
    grad_x0 = 2 * (x(x0, K, r) - x_data) * Sensitivity_x0(x0, K, r, x(x0, K, r))
    grad_K = 2 * (x(x0, K, r) - x_data) * Sensitivity_K(x0, K, r, x(x0, K, r))
    grad_r = 2 * (x(x0, K, r) - x_data) * Sensitivity_r(x0, K, r, x(x0, K, r))
    return np.array([np.sum(grad_x0), np.sum(grad_K), np.sum(grad_r)])
