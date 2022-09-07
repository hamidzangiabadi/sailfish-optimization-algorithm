import numpy as np
import pandas as pd
from numpy.random import rand
from sklearn.model_selection import train_test_split
from sklearn import svm
from copy import deepcopy
import math
import random


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()     
        # X[i,dim-1] = random.uniform(1,35000) # for c
        # X[i,dim-2] = random.uniform(1,32)    # for gamma 
    return X

def sigmoid1(gamma):
    #print(gamma)
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def binary_conversion(X, dim):
    Xbin = np.zeros(dim)
    for d in range(dim):
        if X[d] > 0.5:
            Xbin[d] = 1
        else:
            Xbin[d] = 0
    
    return Xbin


def binary_conversion1(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    
    for i in range(N):
        for d in range(dim - 2):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0

        #for svm params
        Xbin[i,dim-1] = X[i,dim-1]
        Xbin[i,dim-2] = X[i,dim-2]
        # print(X[i,dim-1])

    return Xbin

def global_best( pop):
    minn = 100
    temp = pop[0][1]
    for i in pop:
        #print(i[1])
        minn = min(minn, i[0][1])
        temp = i
    return temp

def _get_global_best__( pop):
    minn = 100
    temp = pop[0]
    # print(temp)
    return temp

# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']



    svm_params = x[len(x)-2:len(x)]
    # print(svm_params)
    x = x[:]

    if(sum(x) == 0):
        return 1

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug   
    # Training
    # print(svm_params[1])
    
    mdl     = svm.SVC()
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred   = mdl.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
    error   = 1 - acc
    
    return error


def accuracy(xtrain, ytrain, x, opts):
    
    dim = np.size(xtrain, 1) 
    x = binary_conversion(x,dim)
    # Parameters
    alpha = 0.6 # related to error
    beta = 0.4  # related to number of features

    svm_params = x[len(x)-2:len(x)]
    features = x[:]
    featuresnp = np.array(features)

    # Original feature size
    max_feat = len(features)
    # Number of selected features
    num_feat = np.sum(features == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = 1 - error
   
    return cost

def generateSVMParams():
    svmParams = (random.uniform(1,35000),random.uniform(1,32))
    return svmParams


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):

    dim = np.size(xtrain, 1) 
    x = binary_conversion(x,dim)
    # Parameters
    alpha = 0.6 # related to error
    beta = 0.4  # related to number of features

    svm_params = x[len(x)-2:len(x)]
    features = x[:]
    featuresnp = np.array(features)

    # Original feature size
    max_feat = len(features)
    # Number of selected features
    num_feat = np.sum(features == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = (alpha * error) + ( beta * (num_feat / max_feat))
   
    return cost


datasetList = ["BreastCancer","Heart","LungCancer","Lymphography","Tictactoe","Zoo","Sonar"]

for dt in datasetList:
    
    # load data
    data  = pd.read_csv("./datasets/"+dt+".csv")
    data  = data.values
    feat  = np.asarray(data[:, 0:-1])
    label = np.asarray(data[:, -1])
    


    # split data into train & validation (70 -- 30)
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)

    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}
    opts = {'fold':fold, 'DT' : dt}
    
    # parameters
    epoch = 50
    pop_size = 70
    pp = 0.1
    A, epxilon = 4, 0.001
    lb = 0
    ub = 1

    ID_POS = 0
    ID_FIT = 1
    ID_SVM_PARAM = 2

    s_size = int(pop_size / pp)

    dim = np.size(xtrain, 1) 
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')


    sf_pop = init_position(lb, ub, pop_size, dim)
    s_pop  = init_position(lb, ub, s_size, dim)

    
    

    ######## for fitness 
    
    sf_fit   = np.array([],dtype=object)
    s_fit    = np.array([],dtype=object)

    svm_param_gamma  = np.array([],dtype=object)
    svm_param_c   = np.array([],dtype=object)
    
    ######## calculate fitness of population

    for i in range(pop_size):
        sf_fit = np.append(sf_fit, Fun(xtrain, ytrain, sf_pop[i,:], opts))
        svm_param_c = np.append(svm_param_c, random.uniform(1,35000)) #c param in svm
        svm_param_gamma = np.append(svm_param_gamma,random.uniform(1,32)) #gamma param in svm

    sf_fits = list(zip(sf_pop, sf_fit,svm_param_c,svm_param_gamma))

    print(sf_fits)
    
    
    for i in range(s_size):
        s_fit = np.append(s_fit, Fun(xtrain, ytrain, s_pop[i,:], opts))
    
    s_fits = list(zip(s_pop, s_fit))

    #### sf_fits = sailfish with fitness matrix
    #### sf_fit = sailfish fitness
    #### s_fits = sardines with fitness matrix
    #### s_fit = sardines fitness
    # g_best = sailfish
    #  s_gbest = sardines

    sf_gbest = global_best(sf_fits)
    s_gbest = global_best(s_fits)

    ################################################## evolve

    nfe_epoch = 0
    pop_new = []
    PD = 2
    AP = 2
    epxilon = 0.0001

    f = open("./outputs/"+dt+".txt", "w")

    for iterno in range(0, epoch):

        print( "----- epoch " + str(iterno))

        for idx in range(0, pop_size):
            lamda_i = 2 * np.random.uniform() * PD - PD
            # print("before")
            # print(sf_fits[idx])
            #Eq. 6
            pos_new = sf_gbest[ID_POS] - lamda_i * ( np.random.uniform() * ( sf_gbest[ID_POS] + s_gbest[ID_POS] ) / 2 - sf_fits[idx][ID_POS] )
            sf_pop_fit = sf_fits[idx][ID_FIT]
            new_tuple = (pos_new, sf_pop_fit)
            sf_fits[idx] = new_tuple
            # print("after")
            # print(sf_fits[idx])

        AP = AP * (1 - 2 * (epoch + 1) * epxilon)

        if AP < 0.5:
            for i in range(0, len(s_fits)):

                temp = (sf_gbest[ID_POS] + AP) / 2
                pos_new = np.random.uniform() * (temp - s_fits[i][ID_POS])
                s_pop_fit = s_fits[i][ID_FIT]
                new_tuple = ( pos_new, s_pop_fit)
                s_fits[i] = new_tuple
        else:
            for i in range(0, len(s_fits)):
                # Eq 9

                pos_new = np.random.uniform() * (sf_gbest[ID_POS] - s_fits[i][ID_POS] + AP)
                s_pop_fit = s_fits[i][ID_FIT]
                new_tuple = ( pos_new, s_pop_fit)
                s_fits[i] = new_tuple

        ## Recalculate the fitness of all sardine
        for i in range(0, len(s_fits)):
            s_pop_arr = s_fits[i][ID_POS]
            s_pop_fit = Fun(xtrain, ytrain, s_fits[i][ID_POS], opts)
            new_tuple = (s_pop_arr, s_pop_fit)
            s_fits[i] = new_tuple

        # for i in range(0, len(sf_fits)):
        #     s_pop_arr = sf_fits[i][ID_POS]
        #     s_pop_fit = Fun(xtrain, ytrain, sf_fits[i][ID_POS], opts)
        #     new_tuple = (s_pop_arr, s_pop_fit)
        #     sf_fits[i] = new_tuple




        # Sort the population of sailfish and sardine (for reducing computational cost)
        sf_fits = sorted(sf_fits, key=lambda temp: temp[ID_FIT])
        s_fits = sorted(s_fits, key=lambda temp: temp[ID_FIT])

    
        for i in range(0, pop_size):
            s_size_2 = len(s_pop)
            if s_size_2 == 0:
                pass
            for j in range(0, s_size):
                ### If there is a better solution in sardine population.
                if sf_fits[i][ID_FIT] > s_fits[j][ID_FIT]: # need to modify
                    sf_fits[i] = deepcopy(s_fits[j])
                    del s_fits[j]
                break   #### This simple keyword helped reducing ton of comparing operation.
                        #### Especially when sardine pop size >> sailfish pop size


        sf_current_best = _get_global_best__(sf_fits)
        s_current_best = _get_global_best__(s_pop)
        if sf_current_best[ID_FIT] < sf_gbest[ID_FIT]:
            sf_gbest = np.array(deepcopy(sf_current_best),dtype=object)
        if s_current_best[ID_FIT] < s_gbest[ID_FIT]:
            s_gbest = np.array(deepcopy(s_current_best),dtype=object)

        
        x = sigmoid(sf_gbest[0])
        fittestSol = sf_gbest
        fittestSol_Accuracy = accuracy(xtrain, ytrain, sf_gbest[ID_POS], opts)
        fittestSol_Fit = Fun(xtrain, ytrain, sf_gbest[ID_POS], opts)
        dim1 = np.size(xtrain, 1) 
        selectedFeaturesSubset = binary_conversion(X=fittestSol[0],dim= dim )
        featuresCount = int(sum(selectedFeaturesSubset))
        print("best subset : " + str(selectedFeaturesSubset))
        print("best accuracy : " + str(fittestSol_Accuracy))
        print("best Fit : " + str(fittestSol_Fit))
        print("features count : " + str(featuresCount))

        f.write(str( iterno + 1)+";"+str(fittestSol_Accuracy)+";"+str(fittestSol_Fit)+";"+ str(featuresCount) +";"+"\n")
        # output in file

