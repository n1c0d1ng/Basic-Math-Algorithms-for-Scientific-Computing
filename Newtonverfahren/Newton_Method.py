import math
import numpy as np

# Konstanten 
#---------------------------------------------------
epsilon = 0.000001
maxiter = 100000

# Newton Verfahren
#----------------------------------------------------
def minimize(f,gradf,H,x0):
    error = 1
    iter = 0
    while error > epsilon:

        # Update
        matrix = H(x0)
        vector = -gradf(x0)

        # Solve Newton Equation
        d = CG(matrix,vector,x0)
        x_neu = x0 + d
        error = norm_vector(gradf(x_neu))
        x0 = x_neu
        iter = iter + 1

        if iter > maxiter:
            break
        
    # Ausgabe der Ergebnisse
    #--------------------------------------------------------------
    print("----------------------------------------------------------------")
    print("Benötigte Iterationen: %d" % (iter))
    print ("Minimalstelle: ")
    print(x_neu)
    print("Funktionswert: %f" % f(x_neu))
    return x_neu


# CG Verfahren zur Lösung eines symmetrischen LGS
#----------------------------------------------------------
def CG(A,b,x_start):
    iter = 0
    r = b - np.dot(A,x_start)
    d = r

    while norm_vector(r) > epsilon:
        # Update der Schrittweite 
        alpha = np.dot(r,d)*(1/np.dot(d,np.dot(A,d)))

        # Update der Iterierten
        x = x_start + np.dot(alpha,d)

        # Update des Residuums
        r = b-np.dot(A,x)

        # Update der Suchrichtung
        beta = np.dot(r,np.dot(A,d))*(1/(np.dot(d,np.dot(A,d))))
        d = r - beta*d

        x_start = x
        iter  = iter + 1

        # Abbruchkriterium
        if iter > maxiter:
            break

    return x

def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)