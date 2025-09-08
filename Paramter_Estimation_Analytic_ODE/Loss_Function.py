# Implementierung der mathematischen Grundlagen des Programms
import numpy as np 

# x' = r x (1- x/K) -> x(t) = f(t) = K / (1 + ((K - x0) / x0) * exp(-r * t))
def f(t, x0, K, r):
    A = (K - x0) / x0
    return K / (1 + A * np.exp(-r * t))

# Ableitung von f nach x0
def df_dx0(t, x0, K, r):
    A = (K - x0) / x0
    return K**2 * np.exp(-r * t) / (x0**2 * (1 + A * np.exp(-r * t))**2)

# Ableitung von f nach K
def df_dK(t, x0, K, r):
    A = (K - x0) / x0
    term1 = 1 / (1 + A * np.exp(-r * t))
    term2 = K * np.exp(-r * t) / (x0 * (1 + A * np.exp(-r * t))**2)
    return term1 - term2

# Ableitung von f nach r
def df_dr(t, x0, K, r):
    A = (K - x0) / x0
    return K * A * t * np.exp(-r * t) / (1 + A * np.exp(-r * t))**2

# LossFunction L = sum (f(t_i) -x_i)^2
def L(x0,K,r, f, x_data, t_data):
    f_vals = f(t_data, x0, K, r)
    return np.sum((f_vals - x_data) ** 2)

# Gradient der LossFunction  wobei L = sum (f(t_i) -x_i)^2
def grad_L(t_data, x_data, x0, K, r):
    f_vals = f(t_data, x0, K, r)
    residuals = f_vals - x_data
    dL_dx0 = 2 * np.sum(residuals * df_dx0(t_data, x0, K, r))
    dL_dK  = 2 * np.sum(residuals * df_dK(t_data, x0, K, r))
    dL_dr  = 2 * np.sum(residuals * df_dr(t_data, x0, K, r))
    return np.array([dL_dx0, dL_dK, dL_dr])