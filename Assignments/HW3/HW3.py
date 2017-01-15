############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
def problem1():    #PCA dimensionality Reduction
    import os
    import numpy as np
    import scipy.io
    from scipy import stats
    import matplotlib.pyplot as plt

    address_wd=os.getcwd()

    data = scipy.io.loadmat(str(address_wd)+'/data 4.mat')
    # print(data)

    X_train=np.asarray(data['X_Question1'])

    D=np.shape(X_train)[0]
    N=np.shape(X_train)[1]
    d=2

    print(N,D)
    X_bar=np.zeros((D,1))
    # print(X_bar)
    for i in range(0,D):
        X_bar[i,0]=np.sum(X_train[i,:])/N

    X_modified=np.zeros((D,N))
    # print(X_bar)

    for i in range(0,N):
        print(i)
        X_modified[:,i]=X_train[:,i]-X_bar[:,0]


    print(X_modified)

    Ux, s, V = np.linalg.svd(X_modified, full_matrices=True)
    U=Ux[:,0:d]
    print (np.shape(U))

    y=np.dot(U.T,(X_modified))
    miu=X_bar
    print(np.shape(y))
    plt.plot(y[0,:],y[1,:],'o')
    plt.show()
    return(miu,U,y)

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
def problem2():    #Kmeans
    import os
    import numpy as np
    import scipy.io
    from scipy import stats
    import matplotlib.pyplot as plt
    address_wd = os.getcwd()
    data = scipy.io.loadmat(str(address_wd) + '/data 4.mat')
    X = np.asarray(data['X_Question2_3'])

    print(np.shape(X))
    (D,N)=np.shape(X)
    K=4
    C=np.random.rand(D, K)

    def distance(a,b):
        return(np.sum((a-b)*(a-b)))


    for t in range(0,1000):
        C_old=np.copy(C)
        print('t = '+str(t))
        Z = np.zeros((N, K))

        for i in range(0,N):
            # print('i = ' + str(i))
            dists=[]
            for k in range(0,K):
                dists.append(distance(X[:,i],C[:,k]))

            nearest=np.argmin(np.array(dists))
            # print(nearest)
            Z[i,nearest]=1


        for j in range(0,K):
            vect=np.zeros((D,1))
            s2=0
            for n in range(0,N):
                vect=vect+np.reshape((X[:,n]*float(Z[n,j])),(D,1))

                s2=s2+float(Z[n,j])
            if s2!=0:
                C[:,j]=list(vect/s2)
            else:
                C[:, j]=list(np.random.rand(D,1))
        # print(np.sum(C))
        print(np.sum((C-C_old)*(C-C_old)))
        if (np.sum((C-C_old)*(C-C_old))==0):
            print('Converged at t =  '+str(t) )
            break


    print(C)
    print(Z)
    plt.plot(X[0,:],X[1,:],'o')         ## Un comment this for plot the whole

    x1_g1=[]
    x2_g1=[]
    x1_g2=[]
    x2_g2 = []
    x1_g3 = []
    x2_g3 = []
    x1_g4 = []
    x2_g4=[]

    for i in range(0,N):
        if Z[i,0]==1:
            x1_g1.append(float(X[0,i]))
            x2_g1.append(float(X[1, i]))
        if Z[i,1]==1:
            x1_g2.append(float(X[0,i]))
            x2_g2.append(float(X[1, i]))
        if Z[i,2]==1:
            x1_g3.append(float(X[0,i]))
            x2_g3.append(float(X[1, i]))
        if Z[i,3]==1:
            x1_g4.append(float(X[0,i]))
            x2_g4.append(float(X[1, i]))
    # plt.plot(x1_g1,x2_g1,'o')
    # plt.plot(x1_g2, x2_g2,'o')
    # plt.plot(x1_g3, x2_g3,'o')
    # plt.plot(x1_g4, x2_g4,'o')

    plt.show()
    return(C,Z)



############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################


############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
def problem3():      #. Spectral Clustering

    import os
    import numpy as np
    import scipy.io
    from sklearn.cluster import KMeans
    import sklearn as sk
    from scipy import stats
    import matplotlib.pyplot as plt

    address_wd = os.getcwd()

    data = scipy.io.loadmat(str(address_wd) + '/data 4.mat')

    X = np.asarray(data['X_Question2_3'])

    print(np.shape(X))

    (d, N) = np.shape(X)

    W=np.zeros((N,N))

    sigma=0.1

    for i in range(0,N):
        for j in range (0,N):
            W[i,j]=np.exp(-np.sum((X[:,i]-X[:,j])**2)/(sigma))
    print('W')
    print(np.shape(W))
    # print(W)

    D=np.zeros((N,N))
    for i in range(0,N):
        D[i,i]=np.sum(W[i,:])
    L=D-W
    # L=np.dot(np.dot(D**(-0.5),L),D**(-0.5))
    # print(L)
    eigenValue,eigenVector=np.linalg.eig(L)

    idx = eigenValue.argsort()[::1]
    eigenValues = eigenValue[idx]

    eigenVectors = eigenVector[:, idx]



    # (M1, M2) = np.shape(eigenVector)
    #
    # idx = np.argpartition(eigenValue, 4)
    # print(idx[:4])
    # print (M1, M2)
    # k = 4
    # print(eigenVector)
    H=eigenVectors[:,0:4]
    print(H)
    print(np.shape(H))


    ####################################### KMEANS PART
    #######################################
    #######################################
    #######################################
    Xnew=H.T
    print(np.shape(Xnew))
    (D, N) = np.shape(Xnew)
    K = 4
    C = np.random.rand(D, K)

    def distance(a, b):
        return (np.sum((a - b) * (a - b)))

    for t in range(0, 500000):
        C_old = np.copy(C)
        print('t = ' + str(t)+' sigma = '+str(sigma))
        Z = np.zeros((N, K))

        for i in range(0, N):
            # print('i = ' + str(i))
            dists = []
            for k in range(0, K):
                dists.append(distance(Xnew[:, i], C[:, k]))

            nearest = np.argmin(np.array(dists))
            # print(nearest)
            Z[i, nearest] = 1

        for j in range(0, K):
            vect = np.zeros((D, 1))
            s2 = 0
            for n in range(0, N):
                vect = vect + np.reshape((Xnew[:, n] * float(Z[n, j])), (D, 1))
                s2 = s2 + float(Z[n, j])

            if s2 != 0:
                # print(list(vect / s2))
                C[:, j] = list(vect / s2)
            else:
                C[:, j] = list(np.random.rand(D, 1))
        # print(np.sum(C))
        print(np.sum((C - C_old) * (C - C_old)))
        if (np.sum((C - C_old) * (C - C_old)) == 0):
            print('Converged at t =  ' + str(t))
            break

    print(C)
    print(Z)
    # plt.plot(X[0,:],X[1,:],'o')         ## Un comment this for plot the whole

    x1_g1 = []
    x2_g1 = []
    x1_g2 = []
    x2_g2 = []
    x1_g3 = []
    x2_g3 = []
    x1_g4 = []
    x2_g4 = []

    for i in range(0, N):
        if Z[i, 0] == 1:
            x1_g1.append(float(X[0, i]))
            x2_g1.append(float(X[1, i]))
        if Z[i, 1] == 1:
            x1_g2.append(float(X[0, i]))
            x2_g2.append(float(X[1, i]))
        if Z[i, 2] == 1:
            x1_g3.append(float(X[0, i]))
            x2_g3.append(float(X[1, i]))
        if Z[i, 3] == 1:
            x1_g4.append(float(X[0, i]))
            x2_g4.append(float(X[1, i]))
    plt.plot(x1_g1, x2_g1, 'o')
    plt.plot(x1_g2, x2_g2, 'o')
    plt.plot(x1_g3, x2_g3, 'o')
    plt.plot(x1_g4, x2_g4, 'o')

    plt.show()
    print(sigma)
    print(t+1)
    return (C, Z)
    ################################### KMEANS 2
    # from sklearn.cluster import KMeans
    # C=KMeans(n_clusters=4).fit(H).labels_
    # x1_g1 = []
    # x2_g1 = []
    # x1_g2 = []
    # x2_g2 = []
    # x1_g3 = []
    # x2_g3 = []
    # x1_g4 = []
    # x2_g4 = []
    # for i in range(0, N):
    #
    #     if C[i] == 0:
    #         x1_g1.append(float(X[0, i]))
    #         x2_g1.append(float(X[1, i]))
    #     if C[i] == 1:
    #         x1_g2.append(float(X[0, i]))
    #         x2_g2.append(float(X[1, i]))
    #     if C[i] == 2:
    #         x1_g3.append(float(X[0, i]))
    #         x2_g3.append(float(X[1, i]))
    #     if C[i] == 3:
    #         x1_g4.append(float(X[0, i]))
    #         x2_g4.append(float(X[1, i]))
    #
    # plt.plot(x1_g1, x2_g1, 'o')
    # plt.plot(x1_g2, x2_g2, 'o')
    # plt.plot(x1_g3, x2_g3, 'o')
    # plt.plot(x1_g4, x2_g4, 'o')
    #
    # plt.show()



############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#Please uncomment anyone of these 3 problems


problem1()
# problem2()
# problem3()

