def FI(X_mat,n):
    m=X_mat.shape[0]
    d1=X_mat.shape[1]
    new_mat=np.ones([m,1])
    for k in range(0,d1):
        for i in range(1,n+1):
            new_mat=np.concatenate((new_mat,X_mat[:,[k]]**i),axis=1)
            # X_mat[:,[k]] used for extracting the k-th column of the X_mat.
    return (new_mat)


def MSE(X,Y,W):
    n=X.shape[0]
    tot_error=0
    Y_model=np.dot(X,W)
    for i in range(0,n):
        error=Y_model[i]-Y[i]
        error=float(error)
        tot_error = error ** 2 + tot_error
    return(tot_error/n)



def accuracy(X,Y,model):
    N=X.shape[0]
    tot_error=0
    # print(model)
    for i in range(0,N):
        # W = np.hstack(W)
        # x_i=np.hstack(X[i])
        # Y_model=np.inner(x_i,W)
        Y_model=(np.dot(X[i],model))
        error=Y_model-Y[i]
        error=float(error)
        tot_error=error**2+tot_error
    return (tot_error/N)

def data_split(X_train,Y_train,K):
    split_dict_data={}
    X_dict={}
    Y_dict={}
    d=X_train.shape[1]
    N = X_train.shape[0]
    all=np.concatenate((X_train,Y_train), axis=1)
    np.random.shuffle(all)  # Shuffle Data
    m=int(N/K)
    for i in range(0,K):
        split_dict_data[i] = all[(i * m):((i * m) + m)]
    for i in range(0,K):
        X_dict[i] = split_dict_data[i][:, 0:d]
        Y_dict[i] = split_dict_data[i][:, d]
    return(X_dict,Y_dict)

def linear_reg(X_train,Y_train):
    x=X_train
    y=Y_train
    W = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x)), x.transpose()), y)
    return(W)

def ridge_regression(X_train,Y_train,Landa):
    x = X_train
    y = Y_train
    W = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x) + Landa * np.eye(d, d)), x.transpose()), y)
    return(W)


def standardize(x):
    n=x.shape[0]
    d=x.shape[1]

    for i in range(0,d):
        column=x[:,i]
        miu=np.mean(column)
        stand_dev=np.std(column)
        print(miu)
        print(stand_dev)
        x[:,i]=(column-miu)/float(stand_dev)
    return (x)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


#Part A Linear Regression:
import numpy as np
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import os
address_wd=os.getcwd()
n_polynomial=2   #Please change n as desiered

data = scipy.io.loadmat(str(address_wd)+'/linear_regression.mat')
X_train=np.asarray(data['X_trn'])
Y_train=np.asarray(data['Y_trn'])
X_test=np.asarray(data['X_tst'])
Y_test=np.asarray(data['Y_tst'])

X_train=FI(X_train,n_polynomial)
X_test=FI(X_test,n_polynomial)
N=X_train.shape[0]
d=X_train.shape[1]
X_train[:,1:d]=stats.zscore(X_train[:,1:d], axis=0, ddof=1)
X_test[:,1:d]=stats.zscore(X_test[:,1:d], axis=0, ddof=1)

x=X_train
y=Y_train
W1=linear_reg(x,y)
print('#########################################################')
print('#########################################################')
print('#########################################################')
print('#########################################################')
print('Linear Regression :')

print('W = ',W1)
print('MSE error of training set = ',MSE(x,y,W1))
print('MSE error of training set = ',MSE(X_test,Y_test,W1))


print('#########################################################')
print('#########################################################')
print('#########################################################')
print('#########################################################')
print('Ridge Regression :')

######################################
######################################
######################################
######################################


# Landa=1
# W2=ridge_regression(x,y,Landa)
K_fold_list=[2,5,10,N]   #You can choose a list
# K_fold_list=[N] # or one value

Landa_list=np.arange(0,1,0.5)

Landa_list=[0,0.1,1,10,20,30,40,50]
Landa_list=np.arange(0,30,0.1)

for K_fold in K_fold_list:
    landa_error_dict={}
    EE_list = []
    for landa in Landa_list:
        X_dict,Y_dict=data_split(X_train,Y_train,K_fold)
        EE = 0

        for i in range(0,K_fold):

            X_holdout=X_dict[i]
            Y_holdout = Y_dict[i]

            if i==0:
                X_tr = X_dict[1]
                Y_tr = Y_dict[1]
                for j in range(2,K_fold):
                    X_tr=np.concatenate((X_tr,  X_dict[j]), axis=0)
                    Y_tr = np.concatenate((Y_tr, Y_dict[j]), axis=0)

            if i!=0:
                X_tr = X_dict[0]
                Y_tr = Y_dict[0]
                for j in range(1,K_fold):
                    if j==i:
                        continue
                    X_tr=np.concatenate((X_tr,  X_dict[j]), axis=0)
                    Y_tr = np.concatenate((Y_tr, Y_dict[j]), axis=0)


            W=ridge_regression(X_tr, Y_tr, landa)
            EE=MSE(X_holdout,Y_holdout,W)+EE
        EE=EE/K_fold
        landa_error_dict[landa]=EE
        EE_list.append(EE)
    if (K_fold==N):
        plt.plot(Landa_list,EE_list)
    plt.show()

    # print(landa_error_dict.keys())
    # print(landa_error_dict.values())
    minimum_landa=min(landa_error_dict, key=landa_error_dict.get)
    print('minimum_landa = ' + str(minimum_landa)+' with K_fold  ='+ str(K_fold ))
    W2 = ridge_regression(x, y, minimum_landa)
    print('MSE error of training set = ',MSE(x, y, W2))
    print('MSE error of test set = ',MSE(X_test, Y_test, W2))
    print('W = ',W2)
    print('#########################################################')
    print('#########################################################')
    print('#########################################################')
    print('#########################################################')
    #######


