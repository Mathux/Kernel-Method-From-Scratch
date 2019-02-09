import pylab as plt
import numpy as np 
from test_data import *
from svm import SVM
from kernel_lr import KernelLogisticRegression
from kernel_knn import KernelKNN

plt.figure(1)
X1,y1,X2,y2 = gen_lin_separable_data()
X = np.vstack([X1,X2])
y = np.hstack([y1,y2])
a = KernelKNN(kernel = 'linear',n_neighbors = 1)
a.fit(X,y)

#plt.scatter(X1[:,0],X1[:,1],color = 'red',marker = 'o')
#plt.scatter(X2[:,0],X2[:,1],color = 'blue',marker = '^')

preds = a.predict(X)

for j in range(len(preds)) :
    if preds[j] == 1 :
        plt.scatter(X[j][0],X[j][1],color = 'magenta',marker = '+')
    elif preds[j] == 0 :
        plt.scatter(X[j][0],X[j][1],color = 'cyan',marker = 'x')
    else : 
        print('pb')
        
#%%
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