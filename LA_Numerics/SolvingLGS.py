# Lösung des entstandenen Gleichungssystems mittels 
# Jacobi Verfahren
#---------------------------------------------------
import numpy as np
import math
epsilon = 0.000001
maxiter = 100


def jacobi_iteration(matrix,vector,x0):
    # Generierung x0
    error = 1
    iter = 0
    n = matrix.shape[0]
    x_neu = np.zeros([n])

    while error > epsilon:
        for i in range(0,n):
            for j in range(0,n):
                if j != i:
                    x_neu[i] = vector[i]-matrix[i][j]*x0[j]
            if matrix[i][i] != 0:
                x_neu[i] = x_neu[i]/matrix[i][i]
        x0 = x_neu
        iter = iter + 1
        if iter > maxiter:
            break
        error = norm_vector(np.dot(matrix,x0)-vector)

    return x_neu


def LR_decomposition(A):
    [n,m] = A.shape 

    # Fehler abfangen bei nicht quadratischer Matrix
    if m != n:
        print("--------------------------------------")
        print("Die Matrix ist nicht quadratisch")
        return

    # LR - Zerlegung
    for k in range(0,m-1):
        A[k+1:m,k] *= 1/A[k,k]
        l = A[k+1:m,k,np.newaxis]
        r = A[np.newaxis,k,k+1:m]
        A[k+1:m,k+1:m] -= np.dot(l,r)

    # Abspeichern der Matrizen
    L = np.identity(m)
    R = np.zeros([m,n])

    for i in range(0,n):
        for j in range(0,m):

            # untere Matrix
            if i>j:
                L[i,j] = A[i,j]

            # obere Matrix
            if j>=i:
                R[i,j] = A[i,j]
    return([L,R])

def CG(A,b,x_start):
    # Initialisiere
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