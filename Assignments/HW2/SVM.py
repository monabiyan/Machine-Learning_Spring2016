import numpy as np
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import os
import random


def plus_minus (Y_train,class_no):

    n=np.shape(Y_train)[0]
    Y_new=np.zeros((n,1))
    for i in range(0,n):
        if (Y_train[i,0]==class_no):
            Y_new[i,0]=1
        else:
            Y_new[i, 0] = -1

    return(Y_new)



def f(alpha,Y,X,x,b):   #output of f should be a single number
    m=np.shape(X)[0]
    sum=0
    for i in range(0,m):
        sum=sum+(alpha[i,0]*Y[i,0]*np.dot(X[i,:],x.T))
    sum=sum+b
    return(sum)


def SMO(C,tol,max_passes,X_train,Y_train):  #implemented from http://cs229.stanford.edu/materials/smo.pdf

    m=np.shape(X_train)[0]  #Number of data points
    d=np.shape(X_train)[1]  #Number of dimensions
    Y = Y_train
    X = X_train
    alpha = np.zeros((m, 1))
    alpha_old=np.zeros((m, 1))
    b=0
    passes=0
    E = np.zeros((m, 1))
    while(passes<max_passes):
        print('passes = ' +str(passes)+ '  E = '+str(np.sum(E)))
        num_changed_alphas=0
        for i in range(0,m):
            E[i,0]=f(alpha,Y,X,X[i,:],b)-Y[i,0]   #output of f should be a single number
            if (((Y[i,0]* E[i,0]<-tol)&(alpha[i,0]<C))|((Y[i,0]* E[i,0]>tol)&(alpha[i,0]>0))):


                indexes=list(range(0,m))
                indexes.remove(i)
                j=random.choice((indexes))
                E[j, 0] = f(alpha, Y, X, X[j,:], b)-Y[j,0]

                alpha_old[i,0]=alpha[i,0]
                alpha_old[j, 0] = alpha[j, 0]

                if (Y[i,0]!=Y[j,0]):
                    L=max(0,alpha[j,0]-alpha[i,0])
                    H=min(C,C+alpha[j,0]-alpha[i,0])
                else :
                    L = max(0, alpha[i, 0] + alpha[j, 0] - C)
                    H = min(C, alpha[i, 0] + alpha[j, 0])


                if (L==H):
                    continue
                eta=float(2*np.dot(X[i,:],X[j,:].T)-np.dot(X[i,:],X[i,:].T)-np.dot(X[j,:],X[j,:].T))
                if ((eta>=0)):
                    continue
                alpha[j,0]=alpha[j,0]-((Y[j,0]*(E[i,0]-E[j,0]))/eta)

                if (alpha[j,0]>H):
                    alpha[j, 0]=H
                elif (L<=alpha[j,0]<=H):
                    alpha[j, 0]=alpha[j, 0]
                else:
                    alpha[j, 0]=L

                if abs(alpha_old[j, 0]-alpha[j, 0])<10**(-5):
                    continue


                alpha[i,0]=alpha[i,0]+Y[i,0]*Y[j,0]*(alpha_old[j,0]-alpha[j,0])

                b1=b-E[i,0]-Y[i,0]*(alpha[i,0]-alpha_old[i,0])*np.dot(X[i,:],X[i,:].T)-Y[j,0]*(alpha[j,0]-alpha_old[j,0])*np.dot(X[i,:],X[j,:].T)

                b2=b-E[j,0]-Y[i,0]*(alpha[i,0]-alpha_old[i,0])*np.dot(X[i,:],X[j,:].T)-Y[j,0]*(alpha[j,0]-alpha_old[j,0])*np.dot(X[j,:],X[j,:].T)

                if (0<alpha[i,0]<C):
                    b=b1
                elif(0<alpha[j,0]<C):
                    b=b2
                else:
                    b=(b1+b2)/2
                num_changed_alphas=num_changed_alphas+1
        if (num_changed_alphas==0):
            passes=passes+1
        else:
            passes=0


    W=np.zeros((1,d))
    for i in range(0, m):
        W=W+alpha[i,0]*Y[i,0]*X[i,:]
    print(W,b)

    return(W,b)



def SVM_Training(X_train,Y_train,C=0.1):

    classes=np.unique(Y_train)
    p=len(classes)    #  p = number of classes

    all_W=[]
    all_b=[]

    for class_no in range(0,p):          #Training for each class

        Y_train_new=plus_minus (Y_train,class_no)
        W,b=SMO(C,0.1,200,X_train,Y_train_new)
        all_W.append(W)
        all_b.append(b)

    model={'all_W':all_W,'all_b':all_b}
    return(model)

def SVM_Testing(X_test,Y_test,model):


    all_W = model['all_W']
    all_b = model['all_b']
    p = len(all_W)  # p = number of classes

    N=np.shape(X_test)[0]
    d=np.shape(X_test)[1]
    correct=0

    Y_out=[]
    for i in range(0,N):
        x=X_test[i,:]
        x=np.reshape(x,(d,1))

        y=Y_test[i,0]
        result=[]

        for class_no in range(0,p):
            outcome_of_SVM=np.dot(all_W[class_no],x) + all_b[class_no]
            result.append(outcome_of_SVM)
        y_model=result.index(max(result))

        print(y_model,y)
        if (y_model ==y):
            correct=correct+1
        Y_out.append(y_model)

    acc_value=correct/N*100
    return(Y_out,acc_value)





address_wd=os.getcwd()

data = scipy.io.loadmat(str(address_wd)+'/data_HW2.mat')
X_train=np.asarray(data['X_trn'])
Y_train=np.asarray(data['Y_trn'])
X_test=np.asarray(data['X_tst'])
Y_test=np.asarray(data['Y_tst'])






def part1():  #PART 1  SMO with linear Kernel

    address_wd = os.getcwd()

    data = scipy.io.loadmat(str(address_wd) + '/data_HW2.mat')
    X_train = np.asarray(data['X_trn'])
    Y_train = np.asarray(data['Y_trn'])
    X_test = np.asarray(data['X_tst'])
    Y_test = np.asarray(data['Y_tst'])



    accuracies = []
    Cs = [1,2,3,4,5,6,7,8,9,10]
    for C in Cs:
        print(C)
        model = SVM_Training(X_train, Y_train, C)
        Y_out, acc = SVM_Testing(X_test, Y_test, model)
        accuracies.append(acc)


    print('############################ ')
    print('SMO accuracy based ')

    print(Cs)
    print(accuracies)

    plt.plot(Cs, accuracies)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.show()


def part2():
    from sklearn import svm

    address_wd = os.getcwd()
    data = scipy.io.loadmat(str(address_wd) + '/data_HW2.mat')
    X_train = np.asarray(data['X_trn'])
    Y_train = np.asarray(data['Y_trn'])
    X_test = np.asarray(data['X_tst'])
    Y_test = np.asarray(data['Y_tst'])


    Kernels=['linear','rbf','poly','sigmoid']
    accuracies=[]
    for K in Kernels:

        clf = svm.SVC(kernel=K,gamma=0.8)
        clf.fit(X_train, Y_train)

        Y_model=clf.predict(X_test)

        n=np.shape(X_test)[0]
        correct=0
        for i in range(0,n):
            if (Y_model[i]==Y_test[i,0]):
                correct=correct+1

        accuracy=correct/n*100
        print(accuracy)
        accuracies.append(accuracy)

    print('############################ ')
    print('SVM Packages Kernel based ')

    print(Kernels)
    print(accuracies)


    plt.plot(accuracies)
    plt.xlabel('Kernels')
    plt.ylabel('accuracy')
    plt.show()



############################  PART 1  SMO with linear Kernel
# part1()


############################  PART 2  SVM from Packages with different Kernels
part2()