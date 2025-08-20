import math
import numpy as np

# Konstanten 
#---------------------------------------------------
epsilon = math.pow(10,-4)
maxiter = math.pow(10,5)
gamma = math.pow(10,-3)
sigma = 0.5


# Quasi Newton Verfahren
# Wir verwenden als H_0 die Einheitsmatrix
#====================================================================
def minimize(J,gradJ,G,gradG,y0,w,f,h,Constraint_number):

    iter = 0

    # Aufbau des Optimierungsvektors
    #--------------------------------------------------------
    N = w.shape[0]
    n = y0.shape[0]
    multiplier = np.zeros((Constraint_number,))
    x0 = np.hstack(( np.hstack((y0,w)),np.zeros((Constraint_number,))))

    # Aufbau der Lagrange-Newton-Matrix
    #------------------------------------------------------
    Lagrange_Newton_Matrix = np.eye(N+n)
    H = Lagrange_Newton_Matrix
    H1 = gradG(y0,w)[:,np.newaxis]
    H2 = np.hstack((H1.T,np.zeros((Constraint_number,Constraint_number))))
    Lagrange_Newton_Matrix = np.hstack((Lagrange_Newton_Matrix,H1))
    Lagrange_Newton_Matrix = np.vstack((Lagrange_Newton_Matrix,H2))
    error = 1


    # Bedingingung H(y0,w) = 0 und Gradient der Lagrange Funktion 0
    # ----------------------------------------------------------------
    while (error > epsilon):

        # Aufbau rechte Seite des Gleichungssystems
        #--------------------------------------------------
        Right_Side = gradJ(y0,w)+x0[-1]*gradG(y0,w)
        Right_Side = (-1)* np.hstack((Right_Side,G(y0,w)))

        search_direction = CG(Lagrange_Newton_Matrix,Right_Side,x0)
        k = 1
        
        # Aufteilen in die komponenten der Suchrichtung
        #----------------------------------------------
        search_y0 = search_direction[:n]
        search_w = search_direction[n:N+n]
        search_multiplier = search_direction[N+n:]

        # Armijo line search
        #---------------------------------------------------------------------------------------------
        while (
            J(y0+math.pow(sigma,k)*search_y0,w+math.pow(sigma,k)*search_w,f,h) - J(y0,w,f,h) > 
            math.pow(sigma,k)*gamma*np.dot(gradJ(y0,w),np.hstack((search_y0,search_w)))
            ):
            k = k+1


        # Update der Iterierten und des aktuellen Fehlers
        # Fehler fällt in der Iteration nicht monoton
        #---------------------------------------------------------------------------------------------
        y0_neu = y0 + math.pow(sigma,k)*search_y0
        w_neu = w + math.pow(sigma,k)*search_w
        multiplier_neu = multiplier + math.pow(sigma,k)*search_multiplier
        error = norm_vector(gradJ(y0_neu,w_neu)+ multiplier*gradG(y0_neu,w))+ abs(G(y0_neu,w_neu))


        # Auswerten der Gradienten Informationen
        #--------------------------------------------------------------------------------------------
        y = gradJ(y0_neu,w_neu)+multiplier*gradG(y0_neu,w_neu) - (gradJ(y0,w)+multiplier*gradG(y0,w))
        z = y[:,np.newaxis]
        Matrix_HS = np.dot(H,search_direction[:N+n])
        Matrix_HS = Matrix_HS[:,np.newaxis]
        

        # Fehler abfangen
        #----------------------------------------------------------
        if (np.dot(y,search_direction[:N+n])< math.pow(10,-9)):
            print("----------------------------------------------------------------")
            print("Nach Iterationen: %d liegt keine invertierbare Matrix mehr vor" % (iter))
            print("Fehler lautet : %f" % error)
            print("Funktionswert: %f" % J(y0,w,f,h) )
            return np.hstack((y0,w))


        # BFGS Update
        # Anmerkung Gradient der Nebenbedingung ist konstant hier, deshalb kein
        # Update notwendig
        #------------------------------------------------------------------------------
        H = H + np.dot(z,z.T)*(1/np.dot(y,search_direction[:N+n]))
        H = H - (np.dot(Matrix_HS,Matrix_HS.T))*(1/(np.dot(search_direction[:N+n],np.dot(H,search_direction[:N+n]))))

        Lagrange_Newton_Matrix = H
        H1 = gradG(y0_neu,w_neu)[:,np.newaxis]

        H2 = np.hstack((H1.T,np.zeros((Constraint_number,Constraint_number))))
        Lagrange_Newton_Matrix = np.hstack((Lagrange_Newton_Matrix,H1))
        Lagrange_Newton_Matrix = np.vstack((Lagrange_Newton_Matrix,H2))


        iter = iter + 1
        if iter > maxiter:
            break
        
        # Abspeichern der Iterierten für die nächste Iteration
        #------------------------------------------------------
        y0 = y0_neu
        w  = w_neu
        multiplier = multiplier_neu
        x0 = np.hstack(( np.hstack((y0,w)),multiplier))

    # Ausgabe der Ergebnisse
    #--------------------------------------------------------------
    print("----------------------------------------------------------------")
    print("Benötigte Iterationen: %d" % (iter))
    print("Startpunkt lautet")
    print(y0)
    print("Funktionswert: %f" % J(y0,w,f,h) )

    return 

# CG Verfahren zur Lösung eines symmetrischen LGS
#====================================================================
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
#====================================================================
def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)