# Beschreibung: Implementierung des steilsten Abstiegs am Beispiel
# der Rosenkopf Funktion
# ----------------------------------------------------------------
import math
import numpy as np
import Gradientenverfahren

# Input der Funktion
def obj_function(x):
    return math.pow(x[0],2)+math.pow(x[1],2)

# Berechnung des Gradienten der Funktion
def gradf(x):
    gradient = np.array([2*x[0],2*x[1]])
    return gradient

# Startwert 
x = np.array([0,7])

# Aufruf des Gradientenverfahrens
Gradientenverfahren.steepestdescent(x,obj_function,gradf)