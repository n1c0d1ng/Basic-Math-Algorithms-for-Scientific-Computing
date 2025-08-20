# Beispiel einer Monte Carlo Simulation
import random 

# Eingabe der Parameter
a = 1
b = 2
n = 1000
I = 0

# Monte Carlo Methode
for i in range(1,n+1):
    # Simulation der Zufallsvariablen mittels Random
    Xk = random.uniform(a, b) 
    Yk = (b-a)*Xk*Xk
    I = I + Yk

# Ausgabe des Wertes
Integral = I/n
print("Der Wert des Integrals lautet I = %2.4f" % Integral)