# Strategie der aktiven Mengen
# Input: Constraints Ax - b <= 0
# Aufstellen des zugehörigen EQP und Lösung von diesem
# =====================================================
import numpy as np
import math
import EQP

# Notwendige Parameter
#------------------------------------------------------
epsilon = 0.00001
maxiter = 5
sigma = 0.5
N = 100000

# Lösung des QPLC
#-----------------------------------------------------
def QPLC(A,b,x0,H,c):

    # Aufstellen der aktive Constraints
    optimality = False
    iter = 0
    QP_dimension  = H.shape[0]
    Constraints = Constraint_Matrix(A,b,x0)
    ConstraintVector = Constraint_Vector(A,b,x0)

    while optimality == False:

        iter = iter + 1
        if iter > maxiter:
            optimality = True

        # Lösung des EQP
        x_EQP = EQP.solve(H,Constraints,ConstraintVector,c,x0)
        x_neu = x_EQP[:QP_dimension]
        x_neu[abs(x_neu)< epsilon] = 0

        # Prüfe ob feasible 
        if Feasible_check(A,x_neu,b) == True:
            # Prüfe ob Lagrange Multiplikatoren >= 0
            if min(x_EQP[QP_dimension:]) > -epsilon:
                print("Der Algorithmus terminiert und hat das Minimum gefunden.")
                print("---------------------------------------------------------\n")
                optimality = True
        
            # Lösche eine Restriktion
            Constraints = Delete_constraint(A,x_EQP,b,Constraints,QP_dimension)
            ConstraintVector = Delete_constraint_Vector(A,x_EQP,b,ConstraintVector,QP_dimension)

        if Feasible_check(A,x_neu,b) == False:
            # Suchrichtung
            d = x_neu -x0
            spacegrid = np.linspace(0,1,N)
            k= 1

            while Feasible_check(A,x_neu,b)==False:
                x_neu = x0 + spacegrid[N-k]*d
                k = k+1

            x_neu[abs(x_neu)< epsilon] = 0
            Constraints = Constraint_Matrix(A,b,x_neu)
            ConstraintVector = Constraint_Vector(A,b,x_neu)

        x0 = x_neu

    return x_neu


# Aufstellen der Matrix mit aktiven Nebenbedingungen
#-------------------------------------------------------
def Constraint_Matrix(A,b,x):
    isVector = True
    ActiveEquation = np.array([])
    EqualityConstraints = False
    if sum(Active_Set_Matrix(A,b,x)) > 0:
        EqualityConstraints = True

    if EqualityConstraints == False:
        print("Keine Aktiven Mengen gefunden")
        return

    if EqualityConstraints == True:
        for i in range(0,A.shape[0]):
            if Active_Set_Matrix(A,b,x)[i] == 1:
                if i==Active_Set_Matrix(A,b,x).index(1):
                    ActiveEquation = A[i,:]
                else:
                    ActiveEquation = np.vstack((ActiveEquation,A[i,:]))
                    isVector = False
        
        # Auffassen als 1xn Matrix
        if isVector == True:
            ActiveEquation =  ActiveEquation[np.newaxis,:]

    return ActiveEquation

# Aufstellen des Vektors mit den aktiven Nebenbedingungen
#-------------------------------------------------------
def Constraint_Vector(A,b,x):
        
    ActiveVector = np.array([])
    EqualityConstraints = False

    if sum(Active_Set_Matrix(A,b,x)) > 0:
        EqualityConstraints = True

    if EqualityConstraints == False:
        print("Keine Aktiven Mengen gefunden")
        return

    if EqualityConstraints == True:
        for i in range(0,A.shape[0]):
            if Active_Set_Matrix(A,b,x)[i] == 1:
                if i==Active_Set_Matrix(A,b,x).index(1):
                    ActiveVector = b[i]
                else:
                    ActiveVector = np.hstack((ActiveVector,b[i]))
        
    return ActiveVector

# Zwischenspeichern Aktive Restriktionen
#-------------------------------------------------------
def Active_Set_Matrix(A,b,x):
    ActiveSets = []
    for i in range(0,A.shape[0]):
        ActiveSets.append(0)

    for i in range(0,A.shape[0]):
        if abs((np.dot(A,x)-b)[i]) < epsilon:
            ActiveSets[i] = 1
    return ActiveSets

# Löschen einer Restriktion
# Prüfe ob ein Lagrange Multiplikator negativ ist 
# und wenn ja lösche die zugehörige Restriktion
#------------------------------------------------------
def Delete_constraint(A,x,b,Constraints,QP_dimension):
    for i in range(QP_dimension,x.shape[0]):
        if x[i] < 0:
            k = i - QP_dimension
            Constraints = np.delete(Constraints,k,0)
            break
    return Constraints


def Delete_constraint_Vector(A,x,b,Constraint_Vector,QP_dimension):
    for i in range(QP_dimension,x.shape[0]):
        if x[i] < 0:
            k = i - QP_dimension
            Constraint_Vector = np.delete(Constraint_Vector,k,0)
            break
    return Constraint_Vector


# Check ob Punkt zulässig ist
#------------------------------------------------------
def Feasible_check(A,x,b):
    if max(np.dot(A,x)-b)<epsilon:
        return True
    else:
        return False