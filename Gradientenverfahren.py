# Funktionen zur Optimierung
import math
import numpy as np

# Impolementierte Parameter:
beta = 0.5
gamma = 0.0001
epsilon = 0.0001
maxit = 5

# Implementierung der Armijo-Schrittweitensteuerung
# -------------------------------------------------------
def armijo(y0,w,d,J,gradJ,f,h,n,N):

    # Konvertierung Matrix in Vektor
    if d.shape[0] == 1:
        d_y0 = d[0][0:n]
        d_w = d[0][n:N+n]

    sigma = 1.0 
    phi = gamma * np.dot(gradJ(y0,w),d.T)[0][0]

     
    while (J(y0+sigma*d_y0,w+sigma*d_w,f,h) > J(y0,w,f,h) + sigma*phi ):
        sigma = beta*sigma
    return sigma

# Step by step Gradientenverfahren
# -------------------------------------------------------
def steepestdescent(y0,w,J,gradJ,h,f):
    iter = 0

    # Konvertierung Matrix in Vektor
    if y0.shape[0] == 1:
        y0 = y0[0]

    n = y0.shape[0]
    N = w.shape[0]
    d = (-1)*gradJ(y0,w)

    # Konvertierung Matrix in Vektor
    if d.shape[0] == 1:
        d = d[0]

    # Aufspalten der Variablen 
    d_y0 = d[0:n]
    d_w = d[n:N]

    while ( norm(np.hstack((d_y0,d_w))) > epsilon):

        # Suchrichtung
        d = -gradJ(y0,w)


        # Schrittweite
        sigma = armijo(y0,w,d,J,gradJ,f,h,n,N) 

        # Aufspalten der Variablen 
        d_y0 = d[0][0:n]
        d_w = d[0][n:N+n]

        # Neue Iterierte
        y0 = y0 + sigma*d_y0
        w = w + sigma*d_w
                
        # Ausgaben
        #----------------------------------------------------------
        print("Iteration N = %d" % iter)
        print("Abstiegsrichtung lautet")
        print(d) 
        print("Armijo Schrittweite lautet: %.12f" % sigma)
        print("Neuer Funktionswert ist")
        print(J(y0,w,f,h))
        print("-------------------------------------------------")

        iter = iter + 1

        if iter > maxit:
            break

# Definition der Norm eines Vektors
# -------------------------------------------------------
def norm(x, k= 2):
    # Konvertierung 1xn Matrix in Vektor
    if x.shape[0] == 1:
        x = x[0]

    n = x.shape[0]
    norm_x = 0
    for i in range(0,n):
        norm_x = norm_x + math.pow(x[i],k)
    norm_x = math.sqrt(norm_x)
    return norm_x