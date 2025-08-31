import numpy as np

# Parameter des OU-Prozesses
lambda_true = 1.5
mu_true = 2.0
sigma_true = 0.5
dt = 0.01
n = 1000

# Simulation der Trajektorie
#np.random.seed(42)
X = np.zeros(n)
X[0] = mu_true
for i in range(1, n):
    drift = -lambda_true * (X[i-1] - mu_true) * dt
    diffusion = sigma_true * np.sqrt(dt) * np.random.randn()
    X[i] = X[i-1] + drift + diffusion

# Least Squares-Schätzung
ΔX = X[1:] - X[:-1]
Y = ΔX / dt
X_reg = X[:-1]

# Design-Matrix für lineare Regression
A = np.vstack([X_reg, np.ones_like(X_reg)]).T
a, b = np.linalg.lstsq(A, Y, rcond=None)[0]

# Rückrechnung der OU-Parameter
lambda_hat = -a
mu_hat = b / lambda_hat

# Volatilitätsschätzung
residuals = Y - (a * X_reg + b)
sigma_hat = np.sqrt(np.mean(residuals**2) * dt)

# Ausgabe
print(f"Geschätzte Parameter:")
print(f"λ ≈ {lambda_hat:.4f}")
print(f"μ ≈ {mu_hat:.4f}")
print(f"σ ≈ {sigma_hat:.4f}")