import math
import numpy as np

# Konstanten 
#---------------------------------------------------
epsilon = math.pow(10,-2)
maxiter = 10000
gamma = 0.001
sigma = 0.5


# Quasi Newton Verfahren
# WIr verwenden als H_0 die Einheitsmatrix
#=====================================================
def minimize(f,gradf,x0):
    iter = 0
    N = x0.shape[0]
    x_start = np.zeros(N)
    H = np.eye(N)
    error = norm_vector(gradf(x0))

    while (error > epsilon):
        s = CG(H,-gradf(x0),x_start)
        x_neu = x0
        k = 1

        # Armijo line search
        #---------------------------------------------------------------------------------------------
        while (f(x_neu+math.pow(sigma,k)*s)-f(x_neu)> math.pow(sigma,k)*gamma*np.dot(gradf(x_neu),s)):
            k = k+1
        #---------------------------------------------------------------------------------------------

        x_neu = x0 + math.pow(sigma,k)*s
        error = norm_vector(gradf(x_neu))

        y = gradf(x_neu) - gradf(x0)
        z = y[:,np.newaxis]

        Matrix_HS = np.dot(H,s)
        Matrix_HS = Matrix_HS[:,np.newaxis]

        if (np.dot(y,s)< math.pow(10,-9)):
            print("----------------------------------------------------------------")
            print("Nach Iterationen: %d liegt keine invertierbare Matrix mehr vor" % (iter))
            print ("Abbruch mit aktueller Iterierten: ")
            print(x_neu)
            print("Gradientennorm lautet : %f" % error)
            print("Funktionswert: %f" % f(x_neu))
            return x_neu

        # BFGS Update
        H = H + np.dot(z,z.T)*(1/np.dot(y,s))
        H = H - (np.dot(Matrix_HS,Matrix_HS.T))*(1/(np.dot(s,np.dot(H,s))))
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

# L2 Norm eines Vekotors
#----------------------------------------------------------
def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)