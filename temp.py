import pylab as plt
import numpy as np 
import test_data
import svm
import utils

plt.figure(1)
X1,y1,X2,y2 = test_data.gen_lin_separable_data()
X = np.vstack([X1,X2])
y = np.hstack([y1,y2])
a = svm.SVM(kernel = 'linear', C = 1/4, gamma = 1/2,offset = 1)
a.fit(X,y)

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

alpha = a.alpha/2

xx,yy = np.linspace(-6,15),np.linspace(-6,15)
XX,YY = np.meshgrid(xx,yy)


l = lambda x,y : sum([alpha[i]*a.kernel(X[i],np.array([x,y])) for i in range(len(X))]) 
Z = []
for xxx in xx : 
    for yyy in yy :
        Z.append(l(xxx,yyy))
        
Z = np.array(Z).reshape((50,50))
plt.contour(XX,YY,Z,0)
#%%

x_train,y_train = load_train(mat = True)
x_test = load_test(mat = True)
train,val,train_labels,val_labels = split_dataset(x_train, y_train)


train_0, train_labels_0, test_0, test_labels_0 = train[0].drop('Id',axis = 1).values , train_labels[0]['Bound'].values, val[0].drop('Id',axis = 1).values , val_labels[0]['Bound'].values

a = KernelLogisticRegression(kernel = 'linear',la = 10**-1)
a.fit(train_0,train_labels_0)

a.score(test_0,test_labels_0)



    