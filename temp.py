import pylab as plt
import numpy as np 
import svm

from test_data import gen_non_lin_separable_data, gen_lin_separable_data,gen_circular_data,gen_lin_separable_overlap_data

plt.figure(1)
X1,y1,X2,y2 = gen_lin_separable_overlap_data()
X = np.vstack([X1,X2])
y = np.hstack([y1,y2]).squeeze()
Xst = np.hstack([X,y.reshape((len(y),1))])
np.random.shuffle(Xst)
X = Xst[:,:-1]
y = Xst[:,-1]
a = svm.SVM(kernel = 'gaussian', C = 2, gamma = 1/4)
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

xx,yy = np.linspace(-2,2),np.linspace(-2,2)
XX,YY = np.meshgrid(xx,yy)


l = lambda x,y : sum([alpha[i]*a.kernel(X[i],np.array([x,y])) for i in range(len(X))]) 
Z = []
for xxx in xx : 
    for yyy in yy :
        Z.append(l(xxx,yyy))
        
Z = np.array(Z).reshape((50,50))
plt.contour(XX,YY,Z,0)
#%%



    