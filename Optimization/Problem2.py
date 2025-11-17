# Unser Problem:
# Gegeben ODE: x'= r*x*(1-x/K) = f(x) 
# min L(theta) = min sum (x_i - f(t_i))^2
# theta = (r,K,x0) Parametervektor
# x_i Messdaten und f(t_i) Modellvorhersage

# Analytisches Setting: f(t) = K/(1+((K-x0)/x0)*exp(-r*t))
# f(t) = K/(1+A*exp(-r*t)) mit A = (K-x0)/x0

import logistic_ODE as Model
import numpy as np

data = np.loadtxt('testdaten.csv', delimiter=',', skiprows=1)
t_data = data[:,0]
x_data = data[:,1]
initial_params = [2.0, 2.0, 12.0]  # Startwerte für x0, r, K

objective_function_analytic = Model.LogisticODE(t_data, x_data, analytic=True)
objective_function_numerical = Model.LogisticODE(t_data, x_data, analytic=False)
analytical_solution = objective_function_analytic.optimize(initial_params)
numerical_solution = objective_function_numerical.optimize(initial_params)

print(f"Geschätzte Parameter (Analytisch): x0={analytical_solution[0]}, r={analytical_solution[1]}, K={analytical_solution[2]}")
print(f"Geschätzte Parameter (Numerisch): x0={numerical_solution[0]}, r={numerical_solution[1]}, K={numerical_solution[2]}")
# Vergleiche die Ergebnisse
print("Unterschiede zwischen analytischer und numerischer Lösung:")
print(f"Loss-Funktion analytisch: {objective_function_analytic.L(*analytical_solution)} - Loss-Funktion numerisch: {objective_function_numerical.L(*numerical_solution)}")
print(f"Loss-Function wahre Paramter: {objective_function_analytic.L(1.0, 0.5, 10.0)}")