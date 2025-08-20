# Beispiel einer Monte Carlo Simulation
import random 
a = 1
b = 2
n = 100000
k = 10
Integral = 0

for k in range(1,k+1):
    I = 0
    for i in range(1,n+1):
        # Simulation der Zufallsvariablen mittels Random
        Xk = random.uniform(a, b) 
        Yk = (b-a)*Xk*Xk
        I = I + Yk
    Integral = Integral + I/n
Integral = Integral/k

# Ausgabe des Wertes
print("Der Wert des Integrals lautet I = %2.4f" % Integral)