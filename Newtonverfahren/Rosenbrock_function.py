# Beschreibung: Implementierung der Rosenbrock Funktion
# und verschiedenen Optimierungsalgorithmen
#----------------------------------------------------------------
import math
import numpy as np
import Newton_Method

# Input der Funktion
#----------------------------------------------------------------
def obj_function(x):
    return 100.0*math.pow(x[1] - math.pow(x[0],2),2) + math.pow(1-x[0],2)

# Berechnung des Gradienten der Funktion
#----------------------------------------------------------------
def gradf(x):
    gradient = np.array(
        [-400.0*(x[1]- math.pow(x[0],2))*x[0]-2*(1-x[0]),
        200*(x[1]-math.pow(x[0],2))
        ])
    return gradient

# Berechnung der Hesse Matrix
#----------------------------------------------------------------
def Hessian(x):
    return np.array(
        [
            [1200*math.pow(x[0],2)-400*x[1]+2,-400*x[0]],
            [-400*x[0],200]
        ]
    )

# Startwert 
x = np.array([-10.2,5.0])

# Aufruf des Verfahrens
Newton_Method.minimize(obj_function,gradf,Hessian,x)