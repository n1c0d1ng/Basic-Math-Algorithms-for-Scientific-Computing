# Lösung des equality constrained Quadratic Programm
# Input: Objective Function
# Equality Constraints 
# Lösung des KKT Systems mittels CG Verfahren
#=======================================================
import numpy as np
import math

# Notwendige Parameter
#------------------------------------------------------
epsilon = 0.000001
maxiter = 10000

# Lösung des resulierenden QP-EC
#---------------------------------------------------------
def solve(H,Constraint_Matrix,Constraint_Vector,c,x0):
    KKT_Matrix = np.hstack((2*H,Constraint_Matrix.T))
    dimension = Constraint_Matrix.shape[0]
    hilfsvariable = np.hstack((Constraint_Matrix,np.zeros([dimension,dimension])))
    KKT_Matrix = np.vstack((KKT_Matrix,hilfsvariable))
    righthandside = np.hstack((-c,Constraint_Vector))

    x0 = np.ones(righthandside.shape[0])
    return CG(KKT_Matrix,righthandside,x0)

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

# Definition der Vektornorm
#--------------------------------------------------------------
def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)
