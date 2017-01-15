import math
import multiprocessing
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import scipy.sparse
import scipy.io
import random
import sklearn

from scipy import stats


########################################################################################
########################################################################################
########################################################################################
########################################################################################
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
#dict = AutoVivification()
########################################################################################
########################################################################################
########################################################################################
########################################################################################
def save_dictionary(path,data):      # a utility function to save a dictionary in hard drive
    print('saving catalog...')
    #open('u.item', encoding="utf-8")
    import json
    with open(path,'w') as outfile:
        json.dump(data, fp=outfile)
    # save to file:
    print(' catalog saved')
########################################################################################
########################################################################################
########################################################################################
########################################################################################
def read_dictionary(path):      # a utility function to load a dictionary from hard drive
    import json
    # load from file:
    g = open(path, 'r')
    print('reading ...')
    try:
        data = json.load(g)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        data = {}
    print('reading finished!')
    return(data)
########################################################################################
########################################################################################
########################################################################################
########################################################################################
def extract_and_save_data():   #This function will read the raw data and will clean them , shuffle them and
                                #seprate data from training and Test data, Finally will save it
    X = []
    Y = []
    file_names = []
    root_path = '/Users/mohsennabian/Datasets/MHEALTHDATASET/mHealth_subject'
    for i in range(1, 11):
        full_path = root_path + str(i) + '.log'
        file_names.append(full_path)
    k = 0
    for i in range(0, 10):
        f = open(file_names[i], 'r')
        whole = f.read()
        sentences = whole.split('\n')
        from random import shuffle
        shuffle(sentences)

        for sentence in sentences:
            k = k + 1
            print(k)
            if (sentence == ' ') | (sentence == ''): continue
            sensors = sentence.split('	')
            sensors = [float(x) for x in sensors]
            if int(sensors[23])==0: continue
            Y.append(int(sensors[23]))
            del sensors[23]
            # sensors.insert(0,1)
            X.append(sensors)


    import numpy as np
    X=np.array(X)
    Y=np.array(Y)
    X= (X - X.min(0)) / X.ptp(0)
    N=len(X)
    X=np.c_[np.ones(N),X]

    X=X.tolist()
    Y=Y.tolist()

    import random
    l = len(X)
    d = int(l * 0.2)
    test_index = random.sample(range(0, l), d)
    train_index = list(set(range(0, l)) - set(test_index))

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for i in train_index:
        X_train.append(X[i])
        Y_train.append(Y[i]-1)

    for i in test_index:
        X_test.append(X[i])
        Y_test.append(Y[i]-1)




    s={}
    s['X_train']=X_train
    s['X_test'] = X_test
    s['Y_train']=Y_train
    s['Y_test']=Y_test
    save_dictionary('/Users/mohsennabian/Datasets/MHEALTHDATASET/my_python_AI_dicts/data_dict.txt',s)

def load_data():    #Here we can load the cleaned data
    import numpy as np
    path='/Users/mohsennabian/Datasets/MHEALTHDATASET/my_python_AI_dicts/data_dict.txt'
    a=read_dictionary(path)

    X_train=a['X_train']
    Y_train=a['Y_train']
    X_test=a['X_test']
    Y_test=a['Y_test']

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    n_train=np.shape(X_train)[0]
    d_train=np.shape(X_train)[1]
    n_test=np.shape(X_test)[0]

    Y_train = np.reshape(Y_train, (n_train, 1))
    Y_test = np.reshape(Y_test, (n_test, 1))


    return(X_train,Y_train,X_test,Y_test)

########################################################################################
########################################################################################
########################################################################################
######################################################################################## KNN Function Starts Here !

def knn_classifier(X_train, Y_train, X_test_one_point, Y_test_one_point, k):  #This is a subfunction for KNN
                                                        #This will returns the most voted label of K-NEAREST-NEIGHBORS
    import math
    import numpy as np

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test_one_point = np.array(X_test_one_point)
    Y_test_one_point = np.array(Y_test_one_point)

    l = len(X_test_one_point)

    m = len(X_train)

    d = 0

    dist_list = []

    for i in range(0, m):
        d = math.sqrt(sum((X_test_one_point - X_train[i]) ** 2))
        dist_list.append(-d)
    dist_list=np.array(dist_list)
    indexes = np.argpartition(dist_list, -k)[-k:]
    neighburs = []
    for i in indexes:
        neighburs.append(Y_train[i])
    from statistics import mode
    return (mode(neighburs))


def Mohsen_knn(X_train, Y_train, X_test, Y_test, k):    #The main function for KNN  O(n^2) Time
    m=len(Y_test)
    Y_pred=[]
    for i in range(0,m):
        print(i)
        Y_pred.append(knn_classifier(X_train, Y_train,X_test[i],  Y_test[i], k))

    from sklearn.metrics import accuracy_score
    print(Y_test)
    print(Y_pred)
    print (accuracy_score( Y_test, Y_pred))
########################################################################################  KNN Function Finishes Here !
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################



########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################  Softmax starts here!

def Train_logistic_regression(X_train,Y_train):

    Y_train = Y_train[:, 0]
    x=X_train
    y=Y_train
    m=np.shape(X_train)[0]
    d=np.shape(X_train)[1]



    L=len(np.unique(y))

    tetas = np.zeros([L,d])
    # lam = 1
    iterations = 1000
    learningRate = 0.1
    landa=10
    Js=[]
    Y_conv=Y_convert(y)
    for i in range(0,iterations):
        z = np.dot(x, np.transpose(tetas))
        Probab=((np.exp(z)).T / np.sum(np.exp(z),axis=1)).T
        J=-1/m*np.sum((Y_conv*np.log(Probab)))
        # loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam / 2) * np.sum(
        #     w * w)  # We then find the loss of the probabilities
        gradian=-1/m*(np.dot(x.T,(Y_conv-Probab)))+landa*tetas.T
        Js.append(J)
        tetas = tetas - (learningRate * gradian.T)
        # print('teta = ',tetas)
        print(' Landa = '+str(landa)+'  J = '+str(J)+" ite = "+str(i))
    # plt.plot(Js)
    # plt.show()
    print(landa)
    return tetas


def Test_logistic_regression(X_test,Y_test,Model):

    Y_test=Y_test[:,0]

    Tetas=Model
    z = np.dot(X_test, Tetas.T)
    Probab = ((np.exp(z)).T / np.sum(np.exp(z), axis=1)).T
    preds = np.argmax(Probab, axis=1)

    accuracy = sum(preds == Y_test) / (float(len(Y_test)))
    return accuracy,preds



def Y_convert(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    sparce_Y= scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    new_Y = np.array(sparce_Y.todense()).T
    return new_Y

########################################################################################  Softmax finishes here!
########################################################################################
########################################################################################
########################################################################################
########################################################################################
######################################################################################## Neural Network Starts


def fix1(y,number_of_classes):  #Classes should start at 0 like [0,1,2,3]
    k=number_of_classes;
    y_a = np.zeros((k, 1))
    y_a[y,0]=1
    return(y_a)

def sigma(z):
    z=1/(1+np.exp(-z))
    return(z)

def softmax(a):
    denum=np.sum(np.exp(a))
    a_new=np.exp(a)/denum
    return(a_new)


def g_derivative(a):
    return(a)


def NN_out(x,W1,b1,W2,b2):

    a1 = x

    Z2 = np.dot(W1, a1) + b1
    a2 = sigma(Z2)
    Z3 = np.dot(W2, a2) + b2
    a3 = softmax(Z3)
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

    accuracy = 100 - (error_count / nn * 100)
    print(accuracy)
    return (Y_model,accuracy)


def Neural_Network_Train(X_train,Y_train,neurons_in_hidden=3,reg_lambda = 0.021):


    N = np.shape(X_train)[0]  # training set size
    d=np.shape(X_train)[1]  #dimension of X

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

    for t in range(0,10):
        diff = 0
        for i in range(0,N):
            x = X_train[i, :]
            x = np.reshape(x, (d, 1))
            y = Y_train[i, 0]
            y_vect = fix1(y, k)  # turn to a vector
            ncol = 1
            a3, Z3, a2, Z2, a1 = NN_out(x, W1, b1, W2, b2)
            dW1 = 0
            dW2 = 0
            db1 = 0
            db2 = 0
            for j in range(0,N):
                print('t='+ str(t)+'   j='+str(j))
                x=X_train[j,:]
                x=np.reshape(x,(d,1))
                y=Y_train[j,0]
                y_vect=fix1(y,k)  #turn to a vector
                ncol=1

                # a3, Z3, a2, Z2,a1=NN_out(x,W1,b1,W2,b2)

                delta3=(a3-y_vect)     #to be fixed (has to be multiplied by g_derivatives)
                # sss=np.sum(delta3)
                # plt.plot(t,sss,'o')
                # plt.pause(0.0000001)
                delta2 = np.dot(W2.T,delta3)*(sigma(Z2)*(1-sigma(Z2)))

                dJ_over_dW2=np.dot(delta3,a2.T)
                dJ_over_db2 = delta3

                dJ_over_dW1 = np.dot(delta2, a1.T)
                dJ_over_db1 = delta2

                dW1 =  dW1 + dJ_over_dW1
                db1 =  db1 + dJ_over_db1

                dW2 =  dW2 + dJ_over_dW2
                db2 =  db2 + dJ_over_db2


            W1 = W1 - epsilon * (dW1/N + reg_lambda*W1)
            b1 = b1 - epsilon * (db1/N)
            W2 = W2 - epsilon * (dW2/N + reg_lambda * W2)
            b2 = b2 - epsilon * (db2/N)

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


########################################################################################  Neural Network Finish
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################  SVM Starts


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
        print('C = '+str(C)+'  passes = ' +str(passes)+ '  E = '+str(np.sum(E)))
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
        W,b=SMO(C,0.1,15,X_train,Y_train_new)
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






########################################################################################  SVM Finish
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
######################################################################################## build in Codes

def build_in_ML_logreg(X_train,Y_train,X_test,Y_test):   #Logistic Regression
    from sklearn import linear_model
    logreg = linear_model.LogisticRegression()
    print('training...')
    logreg.fit(X_train,Y_train)
    print('testing......')
    Y_predict=logreg.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score  .....')
    print (accuracy_score(Y_test, Y_predict))

def build_in_ML_neural_network(X_train,Y_train,X_test,Y_test):  #Neural Network
    from sklearn.neural_network import BernoulliRBM
    from sklearn import linear_model
    from sklearn.pipeline import Pipeline
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM()
    clf = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])
    print('training...')
    clf.fit(X_train, Y_train)
    print('testing......')
    Y_predict = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score  .....')
    print (accuracy_score(Y_test, Y_predict))

def build_in_ML_knn(X_train,Y_train,X_test,Y_test,k):    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    print('training...')
    neigh.fit(X_train,Y_train)
    print('testing......')
    Y_predict=neigh.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('K = '+str(k))
    print('accuracy score  .....')
    print (accuracy_score(Y_test, Y_predict))



def build_in_ML_svm(X_train,Y_train,X_test,Y_test):        #SVM
    from sklearn import svm
    clf = svm.LinearSVC()
    print('training...')
    clf.fit(X_train,Y_train)
    print('testing......')
    Y_predict=clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score  .....')
    print (accuracy_score(Y_test, Y_predict))



def build_in_ML_randomforest(X_train,Y_train,X_test,Y_test):  #RAndom Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    print('training...')
    rf.fit(X_train,Y_train)
    print('testing......')
    Y_predict=rf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score  .....')
    print (accuracy_score(Y_test, Y_predict))

def build_in_ML_naieveBase(X_train,Y_train,X_test,Y_test):  #Naieve Base
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()

    print('training...')
    gnb.fit(X_train,Y_train)
    print('testing......')
    Y_predict=gnb.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score  .....')
    print (accuracy_score(Y_test, Y_predict))

def build_in_ML_decisionTree(X_train,Y_train,X_test,Y_test):  #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()

    print('training...')
    clf.fit(X_train,Y_train)
    print('testing......')
    Y_predict=clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score .....')
    print (accuracy_score(Y_test, Y_predict))


def built_in_ML_Logistic_Reg_Ridge(X_train,Y_train,X_test,Y_test,landa):
    from sklearn.linear_model.ridge import RidgeClassifier
    clf=RidgeClassifier(alpha=landa)
    print('training...')
    clf.fit(X_train,Y_train)
    print('testing......')
    Y_predict = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score .....')
    print(accuracy_score(Y_test, Y_predict))

def built_in_ML_NeuralNetwork(X_train,Y_train,X_test,Y_test,hiddenLayer):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes = (hiddenLayer,))
    print('training...')
    clf.fit(X_train, Y_train)
    print('testing......')
    Y_predict = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('hidden = '+str(hiddenLayer))
    print('accuracy score .....')
    print(accuracy_score(Y_test, Y_predict))

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################  PCA Linear

def PCA_linear(X_train,X_test, RO):
    from sklearn.decomposition import PCA
    dim = np.shape(X_train)[1]
    pca = PCA(n_components=dim)  # Define PCA
    pca.fit(X_train)  # Fit PCA
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)


    variance_ratio = pca.explained_variance_ratio_  # Varience List
    culm_variance_ratio = []
    for i in range(1, dim + 1):
        culm_variance_ratio.append(sum(variance_ratio[0:i]))

    plt.plot(culm_variance_ratio, 'o')
    # plt.show()




    for i in range(0, dim):
        if (culm_variance_ratio[dim - i - 1] < RO):
            toBeDeleted = i - 1
            break

    new_dim = dim - toBeDeleted
    print(RO, new_dim)

    X_train_new = X_train_new [:, 0:new_dim]
    X_test_new = X_test_new[:, 0:new_dim]
    return (X_train_new,X_test_new)


########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

########################################################################################
########################################################################################
########################################################################################  PCA Linear

def PCA_Kernel(X_train,X_test,kernel_name ,deg):

    #Degree for poly kernels. Ignored by other kernels.
    from sklearn.decomposition import KernelPCA


    dim = np.shape(X_train)[1]

    if kernel_name == 'rbf':
        pca = KernelPCA(n_components=14, kernel='rbf', gamma=15)  # Define PCA
    if kernel_name == 'sigmoid':
        pca = KernelPCA(n_components=dim,degree=deg,kernel=kernel_name)  # Define PCA
    pca.fit(X_train)  # Fit PCA
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_test)


    return (X_train_new,X_test_new)


########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

# extract_and_save_data()
X_train,Y_train,X_test,Y_test=load_data()

# X_train=X_train[1:2000,:]
# Y_train=Y_train[1:2000,:]
#
#
# X_test=X_test[1:1000,:]
# Y_test=Y_test[1:1000,:]


print("number of training data =  "+str(len(Y_train)))
print("number  of test data    =  "+str(len(Y_test)))

print(np.shape(X_train))

# print(Y_test)
import time
start_time = time.time()




#####################    Soft Max Mohsen
Model_log_reg=Train_logistic_regression(X_train,Y_train)
accuracy,preds=Test_logistic_regression(X_test,Y_test,Model_log_reg)
print (accuracy)
#####################



######################    KNN Mohsen
Mohsen_knn(X_train, Y_train, X_test[0:20], Y_test[0:20],5)
######################



######################  SVM Mohsen
accuracies = []
Cs = [1000]
for C in Cs:
    print(C)
    model = SVM_Training(X_train, Y_train, C)
    Y_out, acc = SVM_Testing(X_test, Y_test, model)
    accuracies.append(acc)
print(accuracies)
end_time = time.time()
dt=end_time-start_time
print("delta_time = "+str(dt))
###########################################################


#####################  Neural Network Mohsen

accuracy_hidden_layer=[]
hidden_layer=[10]   #chaning hidden layer
for h in hidden_layer :
    model=Neural_Network_Train(X_train,Y_train,neurons_in_hidden=h,reg_lambda = 0.021)  #After Cross Validation : 0.021 was chosen as BEST Landa Value

    y_model,accuracy =Neural_Network_Test(X_test,Y_test,model)

    accuracy_hidden_layer.append(accuracy)
print(accuracy_hidden_layer)
###########################################################


############################################################



built_in_ML_NeuralNetwork(X_train,Y_train,X_test,Y_test,5)
built_in_ML_Logistic_Reg_Ridge(X_train,Y_train,X_test,Y_test,10)
build_in_ML_logreg(X_train,Y_train,X_test,Y_test)
build_in_ML_logreg(X_train,Y_train,X_test,Y_test)
build_in_ML_knn(X_train,Y_train,X_test,Y_test)
build_in_ML_randomforest(X_train,Y_train,X_test,Y_test)
build_in_ML_naieveBase(X_train,Y_train,X_test,Y_test)
build_in_ML_decisionTree(X_train,Y_train,X_test,Y_test)
build_in_ML_svm(X_train,Y_train,X_test,Y_test)

########################  PCA
X_train_new,X_test_new=PCA_linear(X_train,X_test,0.90)
print('Number of Dim  =  '+str(np.shape(X_train_new)[1]))
build_in_ML_logreg(X_train_new,Y_train,X_test_new,Y_test)
build_in_ML_knn(X_train_new,Y_train,X_test_new,Y_test,k=5)
build_in_ML_randomforest(X_train_new,Y_train,X_test_new,Y_test)
build_in_ML_naieveBase(X_train_new,Y_train,X_test_new,Y_test)
build_in_ML_decisionTree(X_train_new,Y_train,X_test_new,Y_test)
build_in_ML_svm(X_train_new,Y_train,X_test_new,Y_test)
built_in_ML_NeuralNetwork(X_train_new,Y_train,X_test_new,Y_test,100)




########################


end_time = time.time()
print('Time = '+ str(end_time -start_time))


