# Beispiel von Monte Carlo Ingeration von 
# int_1^2 x^2 dx

from MC_integration import MonteCarloIntegrator
def f(x):
	return x*x
mc = MonteCarloIntegrator(1, 2, seed = 42)
Integral = mc.integrate(f)

print("Der Wert des Integrals lautet I = %2.6f" % Integral)