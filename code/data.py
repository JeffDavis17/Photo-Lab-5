import numpy as np
from sympy import *
import pandas as pd

# Image coordinates selected from MATLAB Data Cursor
earth_coord = np.array([[4847222.33,620352.25,188],
[4847583.31,620856.67,198],
[4847682.86,620482.8,200],
[4847982.96,620996.75,197],
[4847826.21,620166.07,211],
[4847807.7,620781.3,199],
[4847880.34,620506.75,200],
[4848219.1,620520.57,198],
[4848130.93,620377.87,198]],dtype=float)

# Col row
im_coord = np.array([[1383,9353],
[5371,6452],
[2531,5861],
[6439,3385],
[163,4929],
[4791,4713],
[2671,4151],
[2753,1645],
[1638,2267]],dtype=float)


X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4,X5,Y5,Z5,X6,Y6,Z6,X7,Y7,Z7,X8,Y8,Z8,X9,Y9,Z9 = symbols('X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4,X5,Y5,Z5,X6,Y6,Z6,X7,Y7,Z7,X8,Y8,Z8,X9,Y9,Z9')
x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,f = symbols('x1,y1, x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,f')

ec = np.array([[X1,Y1,Z1],
[X2,Y2,Z2],
[X3,Y3,Z3],
[X4,Y4,Z4],
[X5,Y5,Z5],
[X6,Y6,Z6],
[X7,Y7,Z7],
[X8,Y8,Z8],
[X9,Y9,Z9]])

im_c = np.array([[x1,y1],
[x2,y2],
[x3,y3],
[x4,y4],
[x5,y5],
[x6,y6],
[x7,y7],
[x8,y8],
[x9,y9]])

# Calibration Report:
#f = 101.4e-3 # Focal Length (m)
px = 9e-6 # Pixel Size (m)
pp_x = -0.18e-3

# Image Coordinates [col,row]
col = (im_coord[:,0] - 7500/2 -0.5)*px + pp_x
row = (11500/2 - im_coord[:,1] +0.5)*px
im_coord[:,0] = col
im_coord[:,1] = row

# Symoy Substitution Values
variables = [(X1,earth_coord[0,0]),
(X2,earth_coord[1,0]),
(X3,earth_coord[2,0]),
(X4,earth_coord[3,0]),
(X5,earth_coord[4,0]),
(X6,earth_coord[5,0]),
(X7,earth_coord[6,0]),
(X8,earth_coord[7,0]),
(X9,earth_coord[8,0]),
(Y1,earth_coord[0,1]),
(Y2,earth_coord[1,1]),
(Y3,earth_coord[2,1]),
(Y4,earth_coord[3,1]),
(Y5,earth_coord[4,1]),
(Y6,earth_coord[5,1]),
(Y7,earth_coord[6,1]),
(Y8,earth_coord[7,1]),
(Y9,earth_coord[8,1]),
(Z1,earth_coord[0,2]),
(Z2,earth_coord[1,2]),
(Z3,earth_coord[2,2]),
(Z4,earth_coord[3,2]),
(Z5,earth_coord[4,2]),
(Z6,earth_coord[5,2]),
(Z7,earth_coord[6,2]),
(Z8,earth_coord[7,2]),
(Z9,earth_coord[8,2]),
(x1,im_coord[0,0]),
(x2,im_coord[1,0]),
(x3,im_coord[2,0]),
(x4,im_coord[3,0]),
(x5,im_coord[4,0]),
(x6,im_coord[5,0]),
(x7,im_coord[6,0]),
(x8,im_coord[7,0]),
(x9,im_coord[8,0]),
(y1,im_coord[0,1]),
(y2,im_coord[1,1]),
(y3,im_coord[2,1]),
(y4,im_coord[3,1]),
(y5,im_coord[4,1]),
(y6,im_coord[5,1]),
(y7,im_coord[6,1]),
(y8,im_coord[7,1]),
(y9,im_coord[8,1]),
(f,101.4e-3)
]





