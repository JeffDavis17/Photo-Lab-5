import numpy as np
from sympy import *
from PIL import Image
from data import *
import math as m
from numpy.linalg import inv

# This lab is a Space Restection to uses a non-linear least squares approximation to determine the exterior orientation of a photo
im = Image.open("C:\\Users\\jcdav\\Documents\\GitHub\\Photo-Lab-5\\1038.jpg",'r')

# Flying Height - Solved using points 1 and 2
H = Symbol('H')
dist_12 = (X2 - X1)**2 + (Y2 - Y1)**2
dist_12 = dist_12.subs(variables) # Distance in Earth Coordinates
Height = ((x2*(H-Z2)/f) - (x1*(H-Z1)/f))**2 + ((y2*(H-Z2)/f) - (y1*(H-Z1)/f))**2 - dist_12
Height = Height.subs(variables)
Height = solve(Height,H)
Height = Height[1] # Take Positive Value

# Ground Coordinates from Photo
ground_coord = np.zeros([9,2],dtype=float)
Heights = np.zeros(9,dtype=float)
Heights[:] =Height
xi = np.array(earth_coord[:,0])
yi = np.array(earth_coord[:,1])
hi = np.array(earth_coord[:,2])
Xi = np.zeros(9,dtype=float)
Yi = np.zeros(9,dtype=float)

Xi[:] = xi[:]*(Heights[:] - hi[:])/101.4e-3
Yi[:] = yi[:]*(Heights[:] - hi[:])/101.4e-3

# 2D Conformal Transformation:
a,b,Tx,Ty = symbols('a,b,Tx,Ty')

l = np.array([X1,Y1,X2,Y2,X3,Y3,X4,Y4,X5,Y5,X6,Y6,X7,Y7,X8,Y8,X9,Y9])
l1 = np.zeros(18,dtype=float)
A = np.zeros([18,4],dtype=float)
count = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8]
for i in range(0,18,2):
    l1[i] = l[i].subs(variables)
    j = count[i]

    A[i,0] = Xi[j]
    A[i,1] = -Yi[j]
    A[i,2] = 1
    A[i,3] = 0
    A[i+1,0] = Yi[j]
    A[i+1,1] = Xi[j]
    A[i+1,2] = 0
    A[i+1,3] = 1

obs = inv(A.T@A)@(A.T@l1) # Adjustment

# Adjustment for EO Parameters
#Initial Estimates:
omega = 0
phi = 0
kappa = m.atan2(obs[1],obs[0])
X_l = obs[2]
Y_l = obs[3]
Z_l = Height

o,p,k = symbols('o,p,k')
m11 = cos(p)*cos(k)
m12 = sin(o)*sin(p)*cos(k) + cos(o)*sin(k)
m13 = -cos(o)*sin(p)*cos(k) + sin(o)*sin(k)
m21 = -cos(p)*sin(k)
m22 = -sin(o)*sin(p)*sin(k) + cos(o)*cos(k)
m23 = cos(o)*sin(p)*cos(k) + sin(o)*sin(k)
m31 = sin(p)
m32 = -sin(o)*cos(p)
m33 = cos(o)*cos(p)

# r, s and q 
Xa,Ya,Za,Xl,Yl,Zl = symbols('Xa,Ya,Za,Xl,Yl,Zl')
r = m11*(Xa - Xl) + m12*(Ya-Yl) + m13*(Za-Zl)
s = m21*(Xa - Xl) + m22*(Ya-Yl) + m23*(Za-Zl)
q = m31*(Xa - Xl) + m32*(Ya-Yl) + m33*(Za-Zl)

# Functional Model
F = -f*(r/q)
G = -f*(s/q)
dx = Xa - Xl
dy = Ya - Yl
dz = Za - Zl

# Derivatives
b11 = (f/q**2)*(r*(-m33*dy + m32*dz) - q*(-m13*dy + m12*dz))
b12 = (f/q**2)*(r*(cos(p)*dx + sin(o)*sin(p)*dy - cos(o)*sin(p)*dz) - q*(-sin(p)*cos(k)*dx + sin(o)*cos(p)*cos(k)*dy - cos(o)*cos(p)*cos(k)*dz))
b13 = (-f/q)*(m21*dx + m22*dy + m23*dz)
b14 = (f/q**2)*(r*m31 - q*m11)
b15 = (f/q**2)*(r*m32 - q*m12)
b16 = (f/q**2)*(r*m33 - q*m13)
b21 = (f/q**2)*(s*(-m33*dy + m32*dz) - q*(-m23*dy + m22*dz))
b22 = (f/q**2)*(s*(cos(p)*dx + sin(o)*sin(p)*dy - cos(o)*sin(p)*dz))
b23 = (f/q)*(m11*dx + m12*dy + m13*dz)
b24 = (f/q**2)*(s*m31 - q*m21)
b25 = (f/q**2)*(s*m32 - q*m22)
b26 = (f/q**2)*(s*m33 - q*m23)

# Sub values
initial = [(o,omega),(p,phi),(k,kappa),(Xl,X_l),(Yl,Y_l),(Zl,Z_l)]
vec = [(Xa,X1),(Ya,Y1),(Za,Z1),(Xa,X2),(Ya,Y2),(Za,Z2),(Xa,X3),(Ya,Y3),(Za,Z3),(Xa,X4),(Ya,Y4),(Za,Z4),(Xa,X5),(Ya,Y5),(Za,Z5),(Xa,X6),(Ya,Y6),(Za,Z6),(Xa,X7),(Ya,Y7),(Za,Z7),(Xa,X8),(Ya,Y8),(Za,Z8),(Xa,X9),(Ya,Y9),(Za,Z9)]

num_pts = 9 # Desired No of points
# Design Matrix
A = np.zeros([2*num_pts,6],dtype=float)
count = 0
for j in range(0,2*num_pts,2):
    x_now = np.array([b11,b12,b13,b14,b15,b16])
    y_now = np.array([b21,b22,b23,b24,b25,b26])
    for i in range(0,6):
        x_now[i] = ((x_now[i].subs(vec[count:count+3])).subs(variables)).subs(initial)
        y_now[i] = ((y_now[i].subs(vec[count:count+3])).subs(variables)).subs(initial)
    count+=3
    A[j,:] = x_now[:]
    A[j+1,:] = y_now[:]


# Misclosures: j and k 
#j = xi - xo + f(r/q)

count = 0
counter = 0
misclosure = np.zeros(18,dtype=float)
for i in range(9):
    j = im_coord[i,0] - ((F.subs(initial)).subs(vec[count:count+3])).subs(variables) 
    k = im_coord[i,1] - ((G.subs(initial)).subs(vec[count:count+3])).subs(variables) 
    misclosure[counter] = j
    misclosure[counter+1] = k
    counter+=2
    count+=3
print(misclosure)


# adjustment
dx = inv(A.T@A)@(A.T@misclosure)
print(dx)



# Xnotl,ynotl,znotl are supposed to ebbe at the center of the image???? could use google earth


