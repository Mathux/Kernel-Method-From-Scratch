import pylab as plt
import numpy as np 
import svm

from test_data import gen_lin_separable_data

plt.figure(1)
X1,y1,X2,y2 = gen_lin_separable_data()
X = np.vstack([X1,X2])
y = np.hstack([y1,y2]).squeeze()
#XX = np.hstack([X,y])
#np.random.shuffle(XX)
#X = XX[:,:-1]
#y = XX[:,-1]
#np.random.shuffle(X)
a = svm.SVM(kernel = 'linear', C = 1, offset = 10)
a.fit(X,y)


plt.figure(1)
plt.scatter(X1[:,0],X1[:,1],color = 'red',marker = 'o')
plt.scatter(X2[:,0],X2[:,1],color = 'blue',marker = '^')

preds = a.predict(X)
#
#for j in range(len(preds)) :
#    if preds[j] == 1 :
#        plt.scatter(X[j][0],X[j][1],color = 'magenta',marker = '+')
#    elif preds[j] == 0 :
#        plt.scatter(X[j][0],X[j][1],color = 'cyan',marker = 'x')
#    else : 
#        print('pb')
#        

alpha = a.alpha

xx,yy = np.linspace(-2,12),np.linspace(-2,12)
XX,YY = np.meshgrid(xx,yy)


l = lambda x,y : sum([alpha[i]*a.kernel(X[i],np.array([x,y])) for i in range(len(X))]) 
Z = []
for xxx in xx : 
    for yyy in yy :
        Z.append(l(xxx,yyy))
        
Z = np.array(Z).reshape((50,50))
plt.contour(XX,YY,Z,0)
#%%



    