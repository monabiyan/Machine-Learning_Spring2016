import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import scipy.sparse

import scipy.io



def FI(X_mat,n):
    m=X_mat.shape[0]
    d1=X_mat.shape[1]
    new_mat=np.ones([m,1])
    for k in range(0,d1):
        for i in range(1,n+1):
            new_mat=np.concatenate((new_mat,X_mat[:,[k]]**i),axis=1)
            # X_mat[:,[k]] used for extracting the k-th column of the X_mat.
    return (new_mat)






def Y_convert(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    sparce_Y= scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    new_Y = np.array(sparce_Y.todense()).T
    return new_Y


def Train_logistic_regression(X_train,Y_train):

    x=X_train
    y=Y_train
    m=X_train.shape[0]
    d=X_train.shape[1]
    L=len(np.unique(y))

    tetas = np.zeros([L,d])
    # lam = 1
    iterations = 5000
    learningRate = 0.1
    Js=[]
    Y_conv=Y_convert(y)
    for i in range(0,iterations):
        z = np.dot(x, np.transpose(tetas))
        Probab=((np.exp(z)).T / np.sum(np.exp(z),axis=1)).T
        J=-1/m*np.sum((Y_conv*np.log(Probab)))
        # loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam / 2) * np.sum(
        #     w * w)  # We then find the loss of the probabilities
        gradian=-1/m*(np.dot(x.T,(Y_conv-Probab)))
        Js.append(J)
        tetas = tetas - (learningRate * gradian.T)
        # print('teta = ',tetas)
        print('J = ', J)
    plt.plot(Js)
    plt.show()
    return tetas

def Test_logistic_regression(X_test,Y_test,Model):
    Tetas=Model
    z = np.dot(X_test, Tetas.T)
    Probab = ((np.exp(z)).T / np.sum(np.exp(z), axis=1)).T
    preds = np.argmax(Probab, axis=1)

    accuracy = sum(preds == Y_test) / (float(len(Y_test)))
    return accuracy,preds


def extract_data():
    import os
    address_wd=os.getcwd()
    data = scipy.io.loadmat(address_wd+'/HW1_Data/logistic_regression.mat')
    X_train=np.asarray(data['X_trn'])
    Y_train=np.asarray(data['Y_trn'])
    X_test=np.asarray(data['X_tst'])
    Y_test=np.asarray(data['Y_tst'])
    Y_train=Y_train[:,0]
    Y_test=Y_test[:,0]
    return X_train,Y_train,X_test,Y_test




X_train,Y_train,X_test,Y_test=extract_data()


n_polynomial=2
X_train=FI(X_train,n_polynomial)
X_test=FI(X_test,n_polynomial)



Model=Train_logistic_regression(X_train,Y_train)
accuracy,preds=Test_logistic_regression(X_test,Y_test,Model)

print(accuracy)