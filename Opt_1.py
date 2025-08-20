# Besipiel zum Reduced Dsicretication Approach 
# siehe Gerdts p. 225
#===============================================================
import numpy as np
import math
import EQP
import algorithms
import Gradientenverfahren
import BFGS

# Diskretisierungsparamter 
#---------------------------------------------------
t0 = 0
tf = 1
N = 250
h = (tf-t0)/N
Data.epsilon = math.pow(10,-6)

# Dynamisches System
#--------------------------------------------------
def f(t,y,u):
    dynamics = np.array(
        [y[0]/2 + u,math.pow(y[0],2) + math.pow(u,2)/2]
    )
    return dynamics

def f_y(y,u):
    partial_derivative_y = np.array(
        [
            [0.5, 0],
            [2*y[0], 0]
        ]
    )
    return partial_derivative_y

def f_u(y,u):
    partial_derivative_u = np.array(
        [
            [1],
            [2*u]
        ]
    ) 
    return partial_derivative_u

# Constraints
#---------------------------------------------
def G(y0,w):
    return y0[0]-1

def gradG(y0,w):
    N = w.shape[0]
    initial_state = np.array([1,0])
    control_derivative = np.zeros(N)
    return np.hstack((initial_state,control_derivative))

# Zielfunktion
# mittels Runge-Kutta Ordnung 2
#--------------------------------------------------
def J(y0,w,f,h):
    N = w.shape[0]
    y = algorithms.forward(f,w,h,N,y0,2)
    return y[-1][1] - y[0][1] 


# Steuer- und Zustandsfunktionen
#--------------------------------------------------
w = np.zeros(N)
initial_value = 1
z0 = 5
y0 = np.array(
    [initial_value,z0]
) 


# Berechnung des Gradienten der Zielfunktion
# mittels Euler Ordnung 1
#--------------------------------------------------
def gradJ(y0,w):

    n = y0.shape[0]
    N = w.shape[0]

    S_0 = np.hstack((np.eye(n) ,np.zeros((n,N))))
    S = S_0
    y = algorithms.forward(f,w,h,N,y0,n)

    # Sensitivity Equation
    #-------------------------------------------------------
    for i in range(0,N):
        Du = np.zeros((N+n,1))
        Du[i] = 1
        S = S + h*( np.dot(f_y(y[i],w[i]),S) + np.dot(f_u(y[i],w[i]),Du.T) )

    S_y0 = np.vstack((S[0][0:n],S[1][0:n]))
    S_w  = np.vstack((S[0][n:N+n],S[1][n:N+n]))
    p_y0 = np.array([[0,-1]])
    p_yf = np.array([[0,1]])
    gradient = np.hstack((p_y0 + np.dot(p_yf , S_y0),np.dot( p_yf, S_w)))
    return gradient[0]



BFGS.minimize(J,gradJ,G,gradG,y0,w,f,h,1)