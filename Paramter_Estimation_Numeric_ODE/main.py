import numpy as np
from config import MAX_ITER, MAX_BACKTRACKING , START_X0, START_K, START_R, TOL_GRAD , TOL_L
import LS_Sensitivity_Approach

data = np.loadtxt("testdaten.csv", delimiter=",", skiprows=1)
t = data[:, 0]
x = data[:, 1]

y = np.array([START_X0, START_K, START_R])

# Armijo Schrittweitenbestimmung 
def armijo_step(y, grad, L, f, x_data, t_data, c=1e-4, beta=0.5, alpha_init=1.0):
    ForcedExit = False
    alpha = alpha_init
    iteration = 0
    L_current = L(*y, f, x_data, t_data)
    while True:
        y_new = y - alpha * grad

        # Möglichen Overflow vermeiden durch Projektion 
        y_new[0] = max(y_new[0], 1e-8)
        y_new[1] = max(y_new[1], 1e-8)
        y_new[2] = max(y_new[2], 1e-8)

        L_new = L(*y_new, f, x_data, t_data)
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
        grad = Loss_Function.grad_L(t, x, *y)
        alpha = armijo_step(y, grad, Loss_Function.L, Loss_Function.f, x, t)
        y_new = y - alpha * grad

        # Möglichen Overflow vermeiden durch Projektion 
        y_new[0] = max(y_new[0], 1e-8)
        y_new[1] = max(y_new[1], 1e-8)
        y_new[2] = max(y_new[2], 1e-8)
        
        # Prüfung |L'|< epsilon oder |L_new - L_old| < epsilon
        if np.linalg.norm(grad) < TOL_GRAD or \
           np.abs(Loss_Function.L(*y_new, Loss_Function.f, x, t) \
                   - Loss_Function.L(*y, Loss_Function.f, x, t)) < TOL_L:
            break
        y = y_new
    print(f"Geschätzte Parameter: x0 = {y[0]}, K = {y[1]}, r = {y[2]}")