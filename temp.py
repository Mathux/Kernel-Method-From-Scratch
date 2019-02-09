import pylab as plt
import numpy as np 
from test_data import *
from svm import SVM
from kernel_lr import KernelLogisticRegression

plt.figure(1)
X1,y1,X2,y2 = gen_circular_data()
X = np.vstack([X1,X2])
y = np.hstack([y1,y2])
a = KernelLogisticRegression(kernel = 'gaussian',la = 2,gamma = 3/2)
a.fit(X,y)

plt.scatter(X1[:,0],X1[:,1],color = 'red',marker = 'o')
plt.scatter(X2[:,0],X2[:,1],color = 'blue',marker = '^')

alpha = a.alpha

x,y = np.linspace(-6,6),np.linspace(-6,6)
XX,YY = np.meshgrid(x,y)


l = lambda x,y : sum([alpha[i]*a.kernel(X[i],np.array([x,y])) for i in range(len(X))]) 
Z = []
for xx in x : 
    for yy in y :
        Z.append(l(xx,yy))
        
Z = np.array(Z).reshape((50,50))
plt.contour(XX,YY,Z,0)
#%%