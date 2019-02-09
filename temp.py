import pylab as plt
import numpy as np 
from test_data import *
from svm import SVM

plt.figure(1)
X1,y1,X2,y2 = gen_circular_data()
X = np.vstack([X1,X2])
y = np.hstack([y1,y2])
a = SVM(kernel = 'gaussian')
a.fit(X,y)

plt.scatter(X1[:,0],X1[:,1],color = 'red',marker = 'o')
plt.scatter(X2[:,0],X2[:,1],color = 'blue',marker = '^')

alpha = a.alpha

sv = a.support_vectors
x,y = np.linspace(-3,3),np.linspace(-3,3)
XX,YY = np.meshgrid(x,y)


l = lambda x,y : sum([alpha[i]*a.kernel(X[i],np.array([x,y])) for i in sv])
Z = []
for xx in x : 
    for yy in y :
        Z.append(l(xx,yy))
        
Z = np.array(Z).reshape((50,50))
plt.contour(XX,YY,Z,0)
#%%