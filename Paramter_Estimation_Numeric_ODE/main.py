import numpy as np
from config import MAX_ITER, MAX_BACKTRACKING , START_X0, START_K, START_R, TOL_GRAD , TOL_L
import Computation_Functions as Func 

data = np.loadtxt("testdaten.csv", delimiter=",", skiprows=1)
t_data = data[:, 0]
x_data = data[:, 1]

y = np.array([START_X0, START_K, START_R])

# Armijo Schrittweitenbestimmung 
def armijo_step(y, grad, L, c=1e-4, beta=0.5, alpha_init=1.0):
    ForcedExit = False
    alpha = alpha_init
    iteration = 0
    L_current = L(*y, x_data)
    while True:
        y_new = y - alpha * grad

        # Möglichen Overflow vermeiden durch Projektion 
        y_new[0] = np.clip(y_new[0], 1e-8, 1e2)
        y_new[1] = np.clip(y_new[1], 1e-8, 1e2)
        y_new[2] = np.clip(y_new[2], 1e-8, 1e2)

        L_new = L(*y_new, x_data)
        if L_new <= L_current - c * alpha * np.dot(grad, grad):
            break
        if iteration >= MAX_BACKTRACKING:
            ForcedExit = True
            break
        alpha = alpha * beta
        iteration += 1
    if ForcedExit == True:
        return 0.0
    return alpha

# Gradientenverfahren zur Minimierung von L 
if __name__ == "__main__":
    for i in range(MAX_ITER):
        direction = Func.grad_L(*y, x_data)

        # y: aktueller Parameter Vektor 
        alpha = armijo_step(y, direction, Func.L)
        y_new = y - alpha * direction

        # Möglichen Overflow vermeiden durch Projektion 
        y_new[0] = np.clip(y_new[0], 1e-8, 1e2)
        y_new[1] = np.clip(y_new[1], 1e-8, 1e2)
        y_new[2] = np.clip(y_new[2], 1e-8, 1e2)

        # Prüfung |L'|< epsilon oder |L_new - L_old| < epsilon
        if np.linalg.norm(direction) < TOL_GRAD or \
           np.abs(Func.L(*y_new, x_data) - Func.L(*y, x_data)) < TOL_L:
            break
        y = y_new
    print(f"Geschätzte Parameter: x0 = {y[0]}, K = {y[1]}, r = {y[2]}")