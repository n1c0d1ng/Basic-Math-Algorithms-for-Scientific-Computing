# QR-Zerlegung
#=================================
import math
import numpy as np


# Konstanten 
#------------------------------------------------------------
epsilon = 0.00001
maxIter = 100


# QR- Zerlegung 
# Call by Value Q = np.copy(A) und Call by Reference Q = A 
#------------------------------------------------------------
def decomposition(A):
    Q = np.copy(A) 

    # Orthogonalisieren
    #---------------------------------------------------------
    for i in range(1,A.shape[0]):
        u = np.zeros(A.shape[0])
        for j in range(0,i):
            u = u + (np.dot(A[i],Q[j])/np.dot(Q[j],Q[j]))*Q[j]
        Q[i] = A[i] - u

    # Normalisieren
    #---------------------------------------------------------
    for k in range(0,A.shape[0]):
        Q[k] = (1/norm_vector(Q[k]))*Q[k]
    
    #Q = Q.T
    # Runden der Werte
    # optionaler Schritt
    #---------------------------------------------------------
    #Q[np.abs(Q) < epsilon ] = 0.0

    # Berechnung der Matrix R mittels A = Q*R <-> Q.T*A = R
    #---------------------------------------------------------
    R = np.dot(Q,A.T).T
    R[np.abs(R) < epsilon ] = 0.0
    return (Q,R)
    
# QR-Algorithmus
#------------------------------------------------------------
def algorithm(A):
    V = np.eye(A.shape[0])
    for i in range(0,maxIter):
        (Q,R) = decomposition(A)
        print(Q)
        print("-------------------")
        print(R)
        print("-------------------")
        A_neu = np.dot(R.T,Q.T)
        A = A_neu
        A[np.abs(A) < epsilon ] = 0.0
        V = np.dot(Q,V)
    return (A,V)

# Falls keine Build-in Function verwendet wird
#------------------------------------------------------------
def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)


# Testbeispiel
#------------------------------------------------------------
A = np.array(
    [
        [3.0,2.0,1.0],
        [1.0,2.0,4.0],
        [2.0,-1.0,-2.0]
    ]
)

B = np.array(
    [
        [-1.0, 2.0 , 2.0],
        [0.0 , 4.0 , 5.0],
        [4.0 , -3.0 , 2.0]
    ]
)

X = np.array(
    [
        [3.0 , 0.0 , 0.0],
        [6.0, 2.236067977, 0.0],
        [-2.0 , 4.472135955 , 2.236067977]
    ]
)

M = np.array([[1.0,2.0],[2.0,1.0]])

#print(algorithm(M))

(Q,R) = decomposition(M)
#np.linalg.qr(M) 
#print("--------------------------------")
(D,V) = algorithm(M)
