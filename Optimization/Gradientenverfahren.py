# Funktionen zur Optimierung
import math
import numpy as np

# Impolementierte Parameter:
BETA = 0.5
GAMMA = 0.0001
EPSILON = 0.000001
MAXIT = 100000

# Implementierung der Armijo-Schrittweitensteuerung
# -------------------------------------------------------
def armijo(x,d,f,gradf):
    sigma = 1.0 
    phi = GAMMA*gradf(x).dot(d)
    while (f((x+sigma*d)) > f(x)+ sigma*phi):
        sigma = BETA*sigma
    return sigma

# Implementierung des Gradientenverfahren
# -------------------------------------------------------
def steepestdescent(x,f,gradf):
    d = (-1)*gradf(x)
    N = 0
    x_min = x 
    while (norm(d) > EPSILON):
        d = (-1)*gradf(x_min)
        sigma = armijo(x_min,d,f,gradf) 
        x_min = x_min + sigma*d
        N = N+ 1
        if N > MAXIT:
            break

    print("----------------------------------------------------------------")
    print("Benötigte Iterationen: %d" % (N-1))
    print ("Minimalstelle: ")
    print(x_min)
    print("Funktionswert: %f" % f(x_min))


# Definition der Norm eines Vektors
# -------------------------------------------------------
def norm(x, k= 2):
    n = x.shape[0]
    norm_x = 0
    for i in range(0,n):
        norm_x = norm_x + math.pow(x[i],k)
    norm_x = math.sqrt(norm_x)
    return norm_x

# Step by step Gradientenverfahren
# -------------------------------------------------------
def steepestdescent_iterationen(x,f,gradf):
    maxit = 5500
    d = (-1)*gradf(x)
    N = 0
    x_min = x
    print("Startpunkt ist")
    print(x_min)

    while (norm(d) > e):
        d = -gradf(x_min)
        sigma = armijo(x_min,d,f,gradf) #np.float32(armijo(x_min,d,f,gradf))
        x_min = x_min + sigma*d
                
        # Ausgabe des N-ten Schrittes
        if N>2357:
            print("Iteration N = %d" % N)
            print("Abstiegsrichtung lautet")
            print(d) 
            print("Armijo Schrittweite lautet: %.12f" % sigma)
            print("Neuer Punkt ist")
            print(x_min)
            print("-------------------------------------------------")
        N = N+1
        if N > maxit:
            break

    # Ausgaben
    print("Endpunkt:")
    print(x_min)
    print ("Funktionswert an der Stelle: %f" % f(x_min))