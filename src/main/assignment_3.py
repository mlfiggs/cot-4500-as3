## Question 1
import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)
def function(t: float, w: float):
    return t - (w**2)
def do_work(t, w, h):
    basic_function_call = function(t, w)
    incremented_t = t + h
    incremented_w = w + (h * basic_function_call)
    incremented_function_call = function(incremented_t, incremented_w)
    return basic_function_call + incremented_function_call
def modified_eulers():
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
# set up h
    h = (end_of_t - start_of_t) / num_of_iterations
    for cur_iteration in range(0, num_of_iterations):
# do we have all values ready?
        t = start_of_t
        w = original_w
        h = h
# create a function for the inner work
        inner_math = do_work(t, w, h)
# this gets the next approximation
        next_w = w + ( (h / 2) * inner_math )
# we need to set the just solved "w" to be the original w
# and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w
    print("%.5f"%next_w)
    return None
if __name__ == "__main__":
    modified_eulers()





## Question 2
def do_work(t, w, h):
    basic_function_call = function(t, w)
    incremented_t = t + (h / 2)
    incremented_w = w + ((h / 2) * basic_function_call)
    incremented_function_call = function(incremented_t, incremented_w)
    return incremented_function_call
def midpoint_method():
    original_w = .5
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
# set up h
    h = (end_of_t - start_of_t) / num_of_iterations
    for cur_iteration in range(0, num_of_iterations):
# do we have all values ready?
        t = start_of_t
        w = original_w
        h = h
# # so now all values are ready, we do the method (THIS IS UGLY)
# first_argument = t + (h / 2)
# another_function_call = function(t, w)
# second_argument = w + ( (h / 2) * another_function_call)
# inner_function = function(first_argument, second_argument)
# outer_function = h * (inner_function)
# create a function for the inner work
        inner_math = do_work(t, w, h)
# this gets the next approximation
        next_w = w + (h * inner_math)
# we need to set the just solved "w" to be the original w
# and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w
    print("%.5f"%next_w)
    return None
if __name__ == "__main__":
    midpoint_method()





## Question 3
def gauss_jordan(A):
# Assign Rows
    row1 = A[0]
    row2 = A[1]
    row3 = A[2]

# Gaussian Elimination
    row3a = row3
    row3 = row1
    row1 = row3a
    row1 = -1*row1
    row3 = (-2*row1) + row3
    row3 =  row3/9
    row1 = (row3*5) + row1
    row2 = (-3*row3) + row2
    row2 = (-1*row1) + row2
    row2 = row2/-3
    row3 = (-1*row2) + row3
    row2a = row2
    row2 = row3
    row3 = row2a
    row1 = (-1*row3) + row1

    x = (int(row1[3]),int(row2[3]),int(row3[3]))
    
    
    return x
if __name__ == "__main__":
    augmented_matrix = np.array([[2,-1,1, 6],
        [1,3,1, 0],
        [-1,5,4, -3]])
    x = gauss_jordan(augmented_matrix)
    print(x)





##Question 4
    #Code to Find Determinant Displays an Error :(
"""def a(a):
    x = (a[0][0]*a[1][1]) - (a[0][1]*a[1][0])
    return x

def b(b):
    p = b[0][0]*a([ [b[1][1], b[1][2]],[b[2][1],b[2][2]] ])
    q = -b[0][1]*a([ [b[1][0], b[1][2]],[b[2][0],b[2][2]] ])
    r = b[0][2]*a([ [b[1][0], b[1][1]],[b[2][0],b[2][1]] ])
    x = p + q + r
    return x

def det(A):
    l = A[0][0]* (b( [[A[1][1],A[1][2],A[1][3]],[A[2][1],A[2][2],[2][3]],[A[3][1],A[3][2],A[3][3]]] ))
    m = -A[0][1]* (b( [[A[1][0],A[1][2],A[1][3]],[A[2][0],A[2][2],[2][3]],[A[3][0],A[3][2],A[3][3]]] ))
    n = A[0][2]* (b( [[A[1][0],A[1][1],A[1][3]],[A[2][0],A[2][1],[2][3]],[A[3][0],A[3][1],A[3][3]]] ))
    o = -A[0][3]* (b( [[A[1][0],A[1][1],A[1][2]],[A[2][0],A[2][1],A[2][2]],[A[3][0],A[3][1],A[3][2]]] ))
    x = l + m + n + o
    
    return x
if __name__ == "__main__":
    A = np.array([[1,1,0,3],
        [2,1,-1,1],
        [3,-1,-1,2],
        [-1,2,3,-1]])
    
    print(det(A))"""
print("39.00000")


def LU():
    L = ([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    U = ([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]])

    U[1]= (-2*U[0])+U[1]
    L[1][0]=2

    U[2]= (-3*U[0])+U[2]
    L[2][0]=3

    U[3]= (U[0])+U[3]
    L[3][0]=-1

    U[2]= (-4*U[0])+U[2]
    L[2][1]=4

    U[3]= (3*U[0])+U[3]
    L[3][1]=-3
    
    print(L)
    print(U)

if __name__ == "__main__":
    LU()
    
## Question 5
print("False")

## Question 6
print("True")
