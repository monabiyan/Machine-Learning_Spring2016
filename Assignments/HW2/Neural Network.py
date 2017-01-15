#Part A Linear Regression:
import numpy as np
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import os


def fix1(y,number_of_classes):  #Classes should start at 0 like [0,1,2,3]
    k=number_of_classes;
    y_a = np.zeros((k, 1))
    y_a[y,0]=1
    return(y_a)

def f(z):
    z=1/(1+np.exp(-z))
    return(z)

def g(a):
    denum=np.sum(np.exp(a))
    a_new=np.exp(a)/denum
    return(a_new)


def g_derivative(a):
    return(a)


def NN_out(x,W1,b1,W2,b2):

    a1 = x

    Z2 = np.dot(W1, a1) + b1
    a2 = f(Z2)
    Z3 = np.dot(W2, a2) + b2
    a3 = g(Z3)
    return(a3,Z3,a2,Z2,a1)


def Neural_Network_Test(X_test, Y_test,model):
    W1=model['W1']
    b1=model['b1']
    W2=model['W2']
    b2=model['b2']



    nn = np.shape(X_test)[0]
    dd=np.shape(X_test)[1] #number of features

    error_count = 0
    Y_model=[]
    for i in range(0, nn):
        x = X_test[i, :]
        x = np.reshape(x, (dd, 1))

        y = int(Y_test[i])
        a3, Z3, a2, Z2, a1 = NN_out(x, W1, b1, W2, b2)

        out_model = int(np.argmax(a3))
        Y_model.append(out_model)
        if out_model != y:
            error_count = error_count + 1
            print('error')
        print(y, out_model)

    accuracy = 100 - error_count / nn * 100
    print(accuracy)
    return (Y_model,accuracy)


def Neural_Network_Train(X_train,Y_train,neurons_in_hidden=3,reg_lambda = 0.021):


    N = len(X_train)  # training set size
    d=len(X_train[0,:]) #dimension of X

    classes=np.unique(Y_train)
    k=len(classes)    #  k = number of classes


    m=neurons_in_hidden  # m =number of neurons in hidden layer



    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
      # regularization strength




    #initialize Ws and Bs:
    np.random.seed(0)
    W1 = np.random.randn(m,d) / np.sqrt(d)   #small W1
    b1 = np.zeros((m,1))                    #zero b1
    W2 = np.random.randn(k,m) / np.sqrt(m)   #small W2
    b2 = np.zeros((k,1))                    #zero b2


    error=[]
    iteration=[]

    for t in range(0,1000):
        diff = 0
        for j in range(0,N):
            print('t='+ str(t)+'   j='+str(j))
            x=X_train[j,:]
            x=np.reshape(x,(d,1))
            y=Y_train[j,0]
            y_vect=fix1(y,k)  #turn to a vector
            ncol=1


            a3, Z3, a2, Z2,a1=NN_out(x,W1,b1,W2,b2)


            delta3=(a3-y_vect)     #to be fixed (has to be multiplied by g_derivatives)
            delta2 = np.dot(W2.T,delta3)*(f(Z2)*(1-f(Z2)))

            dJ_over_dW2=np.dot(delta3,a2.T)
            dJ_over_db2 = delta3

            dJ_over_dW1 = np.dot(delta2, a1.T)
            dJ_over_db1 = delta2

            dW1 =  dJ_over_dW1
            db1 =  dJ_over_db1

            dW2 =  dJ_over_dW2
            db2 =  dJ_over_db2


            W1 = W1 - epsilon * (dW1 + reg_lambda*W1)
            b1 = b1 - epsilon * (db1)
            W2 = W2 - epsilon * (dW2 + reg_lambda * W2)
            b2 = b2 - epsilon * (db2)

            diff=np.sum(np.abs(a3-y_vect))+diff
        iteration.append(t)
        error.append(diff/N)


    #################################### Plotting Error Convergence
    # plt.plot(iteration,error)
    # plt.xlabel('iteration')
    # plt.ylabel('Error')
    # plt.show()
    ####################################

    model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}

    return(model)


address_wd=os.getcwd()

data = scipy.io.loadmat(str(address_wd)+'/data_HW2.mat')
X_train=np.asarray(data['X_trn'])
Y_train=np.asarray(data['Y_trn'])
X_test=np.asarray(data['X_tst'])
Y_test=np.asarray(data['Y_tst'])


accuracy_hidden_layer=[]
hidden_layer=[10, 20, 30, 50, 100]
for h in hidden_layer :
    model=Neural_Network_Train(X_train,Y_train,neurons_in_hidden=h,reg_lambda = 0.021)  #After Cross Validation : 0.021 was chosen as BEST Landa Value

    y_model,accuracy =Neural_Network_Test(X_test,Y_test,model)

    accuracy_hidden_layer.append(accuracy)
plt.plot(hidden_layer,accuracy_hidden_layer)
plt.xlabel('number of neurons in hidden Layer')
plt.ylabel('Accuracy')
plt.show()
