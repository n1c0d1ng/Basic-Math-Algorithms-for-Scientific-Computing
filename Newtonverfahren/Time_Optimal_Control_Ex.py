# Zeitoptimales Steuerungsproblem:
#======================================================================================
# min int_0^T 1 + (x^2)/2 + (u^2)/2 dt + M*(x(T) -1)^2
# mit Dynamic x' = u und x(0) = 0 und x(T)=1
#--------------------------------------------------------------------------------------
# Für x(T) =1 fest (und damit M=0) erhalten wir die analytische Lösung:
# x(t) = (e^t - e^(-t))/(e^T - e^(-T)) 
# x'(t) = u 
# T = 2/3
#--------------------------------------------------------------------------------------
# Untersuche nun für M beliebig groß und x(T) ob die Löungen gegen x(T)=1 konvergieren
# p' = -x, x' = u, u = -p liefert ODE 2. Ordnung (x''+x =0) x = A*e^t + B*e^(-t)
# Randbedingung: x(0) = 0, p(T) = M(2x(T) -1), H(x,p) =1+ (1/2) *(x^2 + -p^2) 
# Lösung: B=-A und A = (2Me^T)/(2M+1)e^(2T)-2M+1 und T als Lösung von 1-2A^2=0 (bzw. A = 1/sqrt(2))
#--------------------------------------------------------------------------------------

import math
import numpy as np

#--------------------------------------------------------------------------------------
# Numerische Lösung für die optimale Zeit
from scipy.optimize import fsolve
M = 100

def func(T):
    return (2*M*math.exp(T))/( (2*M+1)*math.exp(2*T) - 2*M+1) -(1/math.sqrt(2))  

def state_x(T):
    A = 2*M*math.exp(T)/( (2*M+1)*math.exp(2*T) - 2*M+1)
    return A*math.exp(T)-A*math.exp(-T)

#--------------------------------------------------------------------------------------
# Lösung
root = fsolve(func,1)
state = state_x(root)


print(root)
print(state)
