# ODE Solver mittels 2 stufigem Runge Kutta
import math
import numpy as np
import Data


# Runge Kutta Verfahren 2 stufig
#----------------------------------------------------
# x_{n+1} = x_{n} + h/2 (k_{1} + k_{2})
# k_{1} = f(t_{n},x_{n})
# k_{2} = f(t_{n}+h,x_{n}+h*k_{1})

def solvingODE(ODE_right_side,initial_value):
    h = Data.discretization
    N = Data.Iterations
    n = initial_value.shape[0]

    ODE_solution_function = np.zeros([N,n])
    ODE_solution_function[0][0:n] = initial_value

    for i in range(0,N-1): 
        k1 = ODE_right_side(i*h,ODE_solution_function[i][0:n])
        k2 = ODE_right_side(i*h+h,ODE_solution_function[i][0:n]+h*k1)
        ODE_solution_function[i+1][0:n]= ODE_solution_function[i][0:n]+ (h/2)*(k1+k2)
    return ODE_solution_function

    

# Sensitivitätsgleichung 
# S' = f_{y} * S und S(0) = 1
#----------------------------------------------------
def solving_Matrix_ODE(Sensitivity_Matrix,state_function):
    N = Data.Iterations
    h = Data.discretization
    n = state_function[0].shape[0]    
    S0 = np.identity(n)
    dimensions = (N,n,n)
    S_grid = np.zeros(dimensions)
    S_grid[0][0:n][0:n] = S0

    for i in range(0,N-1):
        k1 = np.dot(Sensitivity_Matrix(state_function[i][0:n]),S_grid[i][0:n][0:n])
        k2 = np.dot(Sensitivity_Matrix(state_function[i][0:n]),S_grid[i][0:n][0:n]+h*k1)
        S_grid[i+1][0:n][0:n] = S_grid[i][0:n][0:n] + (h/2)*(k1+k2)
    return S_grid


# Newton Verfahren
#----------------------------------------------------
def iteration(ODE_right_side,Sensitivity_Matrix,initial_value,Error_Term,Derivative_Error):
    epsilon = Data.epsilon
    error = 1
    iteration = 0
    x_grid = solvingODE(ODE_right_side,initial_value) 
        
    while error > epsilon:

        vector = (-1)*Error_Term(ODE_right_side,initial_value)
        matrix = Derivative_Error(Sensitivity_Matrix,x_grid)

        # Solve Newton Equation
        d = solveLGS(matrix,vector,initial_value)
        #print(d)
        x_neu = initial_value + d
        initial_value = x_neu

        # Auswertung der ODEs
        x_grid = solvingODE(ODE_right_side,initial_value)

        # Berechnung Norm
        error = norm_vector(Error_Term(ODE_right_side,initial_value))

        iteration = iteration + 1
        if iteration == Data.maximum_iteration:
            print("Maximale Anzahl an Iterationen erreicht")
            break

    print("Anzahl Iterationen = %d" % iteration)
    return x_neu


# Lösung LGS durch Jacobi Verfahren
#---------------------------------------------------
def solveLGS(matrix,vector,initial_value):
    error = 1
    n = matrix.shape[0]
    x_neu = np.zeros([n])

    while error > Data.epsilon:
        for i in range(0,n):
            for j in range(0,n):
                if j != i:
                    x_neu[i] = vector[i]-matrix[i][j]*initial_value[j]
            x_neu[i] = x_neu[i]/matrix[i][i]
        initial_value = x_neu
        error = norm_vector(np.dot(matrix,initial_value)-vector)

    return x_neu

def norm_vector(vector):
    n = vector.shape[0]
    norm = 0
    for i in range(0,n):
        norm = norm + math.pow(vector[i],2)
    return math.sqrt(norm)


