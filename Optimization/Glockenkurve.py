# Beschreibung: Implementierung des steilsten Abstiegs am Beispiel
# der Glockenkurve
# ----------------------------------------------------------------
import math
import numpy as np
import Gradientenverfahren

# Input der Funktion
def obj_function(x):
    return -math.exp(-(math.pow(x[0],2)+math.pow(x[1],2)))

# Berechnung des Gradienten der Funktion
def gradf(x):
    gradient = np.array(
        [
        2*x[0]*math.exp(-(math.pow(x[0],2)+math.pow(x[1],2))),
        2*x[1]*math.exp(-(math.pow(x[0],2)+math.pow(x[1],2))) 
        ])
    return gradient

# Startwert 
x = np.array([-1.20,1])

# Aufruf des Gradientenverfahrens
Gradientenverfahren.steepestdescent_iterationen(x,obj_function,gradf)
#Gradientenverfahren.steepestdescent(x,obj_function,gradf)