# Beispiel eines Quadratischen Optimierungsproblems 
# mit linearer inequality constraint
#====================================================
import math
import numpy as np

# Input Daten des Problems
# min x'Hx + c'x
# s.t. Ax <= b
#-----------------------------------------------------
H = np.array(
    [
        [1, 0],
        [0, 1]
    ]
)

c = np.array(
    [-8 , -6]
)

A = np.array(
    [
        [-1,0],
        [0,-1],
        [1, 1]
    ]
)

b = np.array([ 0 , 0, 5])

# Zuläsiger Startpunk
x_start = np.array([0 , 0])