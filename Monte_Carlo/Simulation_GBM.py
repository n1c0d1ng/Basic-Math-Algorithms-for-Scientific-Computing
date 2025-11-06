# Simulation von dS = mu S dt + sigma S dW
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(20)  # Beliebige Zahl, z.B. 60
generate_Plot = True

# Parameter
mu = 0.5
sigma = 1.0
S0 = 1.0
T = 1.0
N = 10
dt = T / N

# Zeitdiskretisierung
t = np.linspace(0, T, N+1)

# Normalverteilte Zufallsvariablen
Z = np.random.normal(0, 1, N)
dW = np.sqrt(dt) * Z

# Abspeichern aller Simulationen
S_explicit = np.zeros(N+1)
S_explicit[0] = S0
S_euler = np.zeros(N+1)
S_euler[0] = S0
S_milstein = np.zeros(N+1)
S_milstein[0] = S0

# Simulation der Geometrischen Brownschen Bewegung
# S = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
for i in range(1, N+1):
    S_explicit[i] = S_explicit[0] * np.exp((mu - 0.5 * sigma**2) * t[i] + sigma * np.sum(dW[:i]))

# Simulation mittels Euler
# S_i+1 = S_i + mu S_i dt + sigma S_i dW_i
for i in range(1, N+1):
    S_euler[i] = S_euler[i-1] + mu * S_euler[i-1] * dt + sigma * S_euler[i-1] * dW[i-1]

# Simulation mittels Milstein
# S_i+1 = S_i + mu S_i dt + sigma S_i dW_i + 0.5 * sigma^2 * S_i * (dW_i^2 - dt)
for i in range(N):
    S_milstein[i+1] = (
        S_milstein[i]
        + mu * S_milstein[i] * dt
        + sigma * S_milstein[i] * dW[i]
        + (1/2) * sigma *  S_milstein[i] * sigma * (dW[i]**2 - dt)
    )

# Speichern der Ergebnisse in eine CSV-Datei
data = np.column_stack((t, S_explicit, S_euler, S_milstein))
np.savetxt("pfade.csv", data, delimiter=",", header="t,Explizit,Euler,Milstein", comments='')

# Aufbauen des Plots
if generate_Plot == True:
    plt.plot(t, S_euler,color = 'blue' ,label='Euler-Maruyama')
    plt.plot(t, S_milstein,color = 'orange' ,label='Milstein')
    plt.plot(t, S_explicit,color = 'red' ,label='Explizite Lösung')
    #plt.xlabel('Zeit')
    #plt.ylabel('S(t)')
    plt.xlim(0, 1.0)
    plt.ylim(0, 3.8)
    plt.show()