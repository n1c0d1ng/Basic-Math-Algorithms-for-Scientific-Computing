import math
import numpy as np

# Konstanten 
#---------------------------------------------------
epsilon = 0.001
maxiter = 100000

# Definition der Daten
#---------------------------------------------------
A = 0.5*np.array(
    [
        [2,-1,0],
        [-1,2,0],
        [0,0,2]
    ]
)

b = np.array(
    [1,1,1]
)


x0 = np.array(
    [1,0,1]
)


# CG Verfahren zur Lösung eines symmetrischen LGS
#---------------------------------------------------
def CG(A,b,x_start):
    iter = 0
    r = b - np.dot(A,x_start)
    d = r
    print(d)

    while norm_vector(r) > epsilon:
        # Update der Schrittweite 
        alpha = np.dot(r,d)*(1/np.dot(d,np.dot(A,d)))
        print(alpha)

        # Update der Iterierten
        x = x_start + np.dot(alpha,d)
        print(x)


        # Update des Residuums
        r = b-np.dot(A,x)

        # Update der Suchrichtung
        beta = np.dot(r,np.dot(A,d))*(1/(np.dot(d,np.dot(A,d))))
        d = r - beta*d
        print(d)
        print("-------------------------------")

        x_start = x
        iter  = iter + 1

        # Abbruchkriterium
        if iter > maxiter:
            break

    return x

# Falls keine Build-in Function verwendet wird
#---------------------------------------------------
def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)


# Lösung des Gleichungssystems
x = CG(A,b,x0)
print(x)
print(np.dot(A,x))
