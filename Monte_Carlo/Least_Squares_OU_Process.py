import numpy as np

# Parameter des OU-Prozesses
lambda_true = 1.5
mu_true = 2.0
sigma_true = 0.5
x0 = 1.5
dt = 0.01
n = 10000

# Simulation von dX = -lambda (X - mu) dt + sigma dW
np.random.seed(42)
X = np.zeros(n)
X[0] = x0
for i in range(1, n):
    drift = -lambda_true * (X[i-1] - mu_true) * dt
    diffusion = sigma_true * np.sqrt(dt) * np.random.randn()
    X[i] = X[i-1] + drift + diffusion

# X[1:] = [x1 x2 .. x(n-1)] x[:-1] = [x0 x1 .. xn]
dX = X[1:] - X[:-1]
Y = dX / dt     # Eine Komponente weniger als X
X_reg = X[:-1]  #Länge passt zu Y
A = np.column_stack([X_reg, np.ones_like(X_reg)]) # A= (x 1)
# Lösen des Normalengleichungssystems
a, b = np.linalg.solve(A.T @ A, A.T @ Y)  

# Rückrechnung der OU-Parameter
lambda_hat = -a
mu_hat = b / lambda_hat

# Ausgabe
print(f"Geschätzte Parameter:")
print(f"λ ≈ {lambda_hat:.4f}")
print(f"μ ≈ {mu_hat:.4f}")