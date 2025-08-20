# Full Discretization Approach for a LQR Problem
# Example Gerdts p. 225
#====================================================

import math
import numpy as np
import EQP
import EulerODE
import matplotlib.pyplot as plt

# Diskretisierungsparamter 
#---------------------------------------------------
t0 = 0
tf = 1
N = 5
h = (tf-t0)/N
epsilon = 0.0000001
x0 = 1

# Dynamisches System
#--------------------------------------------------
def f(x,u):
    return 0.5*x+u

# Diskretsierung des Zielfunktionals
#---------------------------------------------------
A = (h/2)*np.eye(N-1)
print(A)
B = h*np.eye(N)

print(B)
H = np.hstack(
    (
        np.vstack(
            (A,np.zeros([N,N-1]))
        ),
        np.vstack(
            (np.zeros([N-1,N]),B)
        ) 
    )
)

print(H)

c = np.zeros(2*N-1)

# Diskretisierung der ODE mittels explizitem Euler 
#---------------------------------------------------
X = np.diag((-1-h/2)*np.ones(N-1))
print((-1-h/2)*np.ones(N-1))
print(X)

X = np.hstack((X,np.zeros([N-1,1])))
U = -h*np.eye(N-1)
for i in range(0,N-1):
    X[i,i+1] = 1



# Diskretisierung der Anfangsbedingung
#--------------------------------------------------
Initial_Condition = np.zeros((1,2*N-1))
Initial_Condition[0,N-1] = 1

# Erstellen der Constraint Matrix
#-------------------------------------------------
Constraint_matrix = np.hstack((U,X))
Constraint_matrix[abs(Constraint_matrix) < epsilon] = 0
Constraint_matrix = np.vstack((Constraint_matrix,Initial_Condition))
Constraint_vector = np.zeros(N)
Constraint_vector[N-1] = x0 

# Zulässiger Startvektor mittels explizitem Euler
# -------------------------------------------------
u_start = np.zeros([N-1,1])
x_start = EulerODE.forward(f,u_start,h,N,x0,1)
initial_solution = np.vstack((u_start,x_start)).reshape(2*N-1,)


# Lösung des QP
# --------------------------------------------------
solution_EQP = EQP.solve(H,Constraint_matrix,Constraint_vector,c,initial_solution)
u_solution = solution_EQP[:N-1]
time_grid_u = np.linspace(0,1,N-1)
x_solution = solution_EQP[N-1:2*N-1]
time_grid_x = np.linspace(0,1,N)
print(x_solution[-1])

# Ausgabe der Ergebnisse
# --------------------------------------------------
print("Der Wert des Zielfunktionals lautet")
print("-------------------------------------------")
print(np.dot(solution_EQP[:2*N-1],np.dot(H,solution_EQP[:2*N-1]))+np.dot(c,solution_EQP[:2*N-1]))

plt.plot(time_grid_u,u_solution,label='Steuerung')
plt.plot(time_grid_x,x_solution,label='Zustand')
plt.legend(loc='lower right')
plt.title('Graphische Darstellung')
#plt.show()