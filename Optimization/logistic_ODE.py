import numpy as np

MAX_ITER=int(1e5)
MAX_LINESARCH = int(1e2)
TOL_L=1e-4
TOL_GRAD=1e-4
MIN_STEP_SIZE=1e-4
EPSILON = 1e-8


class LogisticODE:
    def __init__(self, t_data, x_data, analytic = False):
        self.t_data = t_data
        self.x_data = x_data
        self.Num_Steps = len(t_data)
        self.h = t_data[1] - t_data[0]
        self.use_analytic = analytic

    def x_analytic(self, x0, r, K):
        A = np.clip((K - x0) / x0, 1e-1, 1e3)
        exp_arg = np.clip(-r*self.t_data, -1e3, 1e3)
        return K/((1 + A*np.exp(exp_arg))+ EPSILON)
    
    def dx_dx0(self,x0,r,K):
        A = np.clip((K - x0) / x0, 1e-1, 1e3)
        # Clippen der Exponentialfunktion um Overflow zu vermeiden
        exp_arg = np.clip(-r*self.t_data, 1e-3, 500.0)
        return (K**2 * np.exp(exp_arg)) / (x0**2 * (1 + A * np.exp(exp_arg))**2 + EPSILON)

    def dx_dr(self,x0,r,K):
        A = np.clip((K - x0) / x0, 1e-1, 1e3)
        exp_arg = np.clip(-r*self.t_data, 1e-3, 500.0)
        return K * A * self.t_data * np.exp(exp_arg) / ((1 + A * np.exp(exp_arg))**2 + EPSILON)

    def dx_dK(self,x0,r,K):
        A = np.clip((K - x0) / x0, 1e-1, 1e3)
        exp_arg = np.clip(-r*self.t_data, 1e-3, 500.0)
        term1 = 1/(1 + A * np.exp(exp_arg))
        term2 = K * np.exp(exp_arg) / (x0 * ((1 + A * np.exp(exp_arg))**2 + EPSILON))
        return term1 - term2

    def x_simulate(self, x0, r, K):
        x_simulated = np.zeros(self.Num_Steps)
        x_simulated[0] = x0 

        # Euler Verfahren
        # x' = r*x*(1-x/K) -> x_{n+1} = x_n + r*x_n*(1-x_n/K)*h
        for i in range(1, self.Num_Steps):
            x_simulated[i] = x_simulated[i-1] + r * x_simulated[i-1] * (1 - x_simulated[i-1]/K) * self.h
            x_simulated[i] = np.clip(x_simulated[i], -50.0, 50.0) 
        return x_simulated
    
    def S_x0(self,x0, r, K, x_grid):
        S_x0 = np.zeros(self.Num_Steps)
        S_x0[0] = 1.0

        # Euler Verfahren 
        for i in range(1, self.Num_Steps):
            S_x0[i] = S_x0[i-1] + (-2*r/K * x_grid[i-1] +r) * S_x0[i-1] * self.h
        return S_x0

    def S_r(self,x0, r, K, x_grid):
        S_r = np.zeros(self.Num_Steps)
        S_r[0] = 0.0
        # Euler Verfahren 
        for i in range(1, self.Num_Steps):
            S_r[i] = S_r[i-1] + (-2*r/K * x_grid[i-1] +r) * S_r[i-1] * self.h + x_grid[i-1] * (1 - x_grid[i-1]/K) * self.h
        return S_r

    def S_K(self,x0, r, K, x_grid):
        S_K = np.zeros(self.Num_Steps)
        S_K[0] = 0.0
        # Euler Verfahren 
        for i in range(1, self.Num_Steps):
            S_K[i] = S_K[i-1] + (-2*r/K * x_grid[i-1] +r) * S_K[i-1] * self.h + r/K**2 * x_grid[i-1]**2 * self.h
        return S_K
    
    # Compute Loss-Function L
    def L(self, x0,r,K):
        if self.use_analytic:
            x_values = self.x_analytic(x0, r, K)
        else:
            x_values = self.x_simulate(x0, r, K)
        return np.sum((self.x_data - x_values)**2)
    
    def gradL(self, x0,r,K):
        if self.use_analytic == True:
            x_values = self.x_analytic(x0, r, K) 
        else:
            x_values = self.x_simulate(x0, r, K)

        residuals = self.x_data - x_values

        if self.use_analytic == True:
            dL_dx0 = -2 * np.sum(residuals * self.dx_dx0(x0, r, K))
            dL_dK = -2 * np.sum(residuals * self.dx_dK(x0, r, K))
            dL_dr = -2 * np.sum(residuals * self.dx_dr(x0, r, K))
        else:
            dL_dx0 = -2 * np.sum(residuals * self.S_x0(x0, r, K, x_values))
            dL_dr = -2 * np.sum(residuals * self.S_r(x0, r, K, x_values))
            dL_dK = -2 * np.sum(residuals * self.S_K(x0, r, K, x_values))
        return np.array([dL_dx0, dL_dr, dL_dK])
    
    def optimize(self, initial_params):
        theta = initial_params.copy()
        for i in range(MAX_ITER):
            grad = self.gradL(*theta) #*y = Entpacken der Parameter
            #Schrittweiten Steuerung mit Armijo
            alpha = self.armijo_step(theta, grad, self.L)
            if np.linalg.norm(grad) < TOL_GRAD:
                break
            theta_new = theta - alpha * grad
            # Clip parameters to valid ranges
            theta_new[0] = np.clip(theta_new[0], 1e-1, 1e2)  # x0
            theta_new[1] = np.clip(theta_new[1], 1e-1, 1e2)  # r
            theta_new[2] = np.clip(theta_new[2], 1e-1, 1e2)  # K
            if np.abs(self.L(*theta_new) - self.L(*theta)) < TOL_L:
                break
            theta = theta_new
        return theta

    
    # Armijo Bedingung: L(theta_new) <= L(theta) - c * alpha * ||gradL(theta)||^2als
    def armijo_step(self, theta,grad, L, alpha_init =1.0,c=1e-4, beta=0.5):
        Forced_Exit = False
        alpha = alpha_init
        iteration = 0
        L_current = L(*theta)
        # Jetzt schrittweitensuche
        while True:
            theta_new = theta - alpha * grad
            L_new = L(*theta_new)
            if L_new < L_current - c * alpha * np.linalg.norm(grad)**2:
                break
            if iteration > MAX_LINESARCH:
                Forced_Exit = True
                break
            alpha = alpha * beta 
            iteration = iteration + 1
            
        if Forced_Exit == True:
            alpha = MIN_STEP_SIZE
        return alpha