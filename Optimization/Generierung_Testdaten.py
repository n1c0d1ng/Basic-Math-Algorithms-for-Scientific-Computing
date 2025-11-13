# Generierung von testdaten 
import numpy as np

def logistische_funktion(t, r, K, x0):
    A = (K - x0) / x0
    return K / (1 + A * np.exp(-r * t))

num_test_data = 100
#Parameter
x0 , K , r = 1, 10, 0.5 

# Zeitgitter anlegen 
t_data = np.linspace(0,10, num_test_data) # 0 bis 1 mit 10 testdaten 
x = logistische_funktion(t_data, r, K, x0)

# Verrauschen der Daten 
x_data = x + np.random.normal(0,0.5, num_test_data) #Normalverteilung mit Erwartungswert 0 und std 0.5

# Abspeichern der testdaten 
data = np.column_stack((t_data, x_data))
np.savetxt('testdaten.csv', data, delimiter=',', header='t_data,x_data', comments='')