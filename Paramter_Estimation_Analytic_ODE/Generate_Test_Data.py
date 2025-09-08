import numpy as np

def logistic_solution(t, x0, K, r):
    return K / (1 + ((K - x0) / x0) * np.exp(-r * t))

# Anzahl Datenpunkte
num_data_points = 100

# Parameter
x0, K, r = 1, 10, 0.5
t = np.linspace(0, 10, num_data_points)
x = logistic_solution(t, x0, K, r)
# Optional: Rauschen hinzufügen
x_noisy = x + np.random.normal(0, 0.2, size=x.shape)
#x_noisy = x

# Speichern der Daten
data = np.column_stack((t, x_noisy))
np.savetxt("testdaten.csv", data, delimiter=",", header="t,x", comments='')