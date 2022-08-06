from telnetlib import SGA
from joblib import PrintTime
import numpy as np
import pandas as pd
from numpy.random import rand
from sklearn.model_selection import train_test_split
from sklearn import svm
from copy import deepcopy


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()     
        # X[i,dim-1] = random.uniform(1,35000) # for c
        # X[i,dim-2] = random.uniform(1,32)    # for gamma 
    return X

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

def global_best( pop):
    minn = 100
    temp = pop[0][1]
    for i in pop:
        #print(i[1])
        minn = min(minn, i[0][1])
        temp = i
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


datasetList = ["Amphibians"]

for dt in datasetList:
    
    # load data
    data  = pd.read_csv(dt+".csv")
    data  = data.values
    feat  = np.asarray(data[:, 0:-1])
    label = np.asarray(data[:, -1])
    


    # split data into train & validation (70 -- 30)
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)

    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}
    opts = {'fold':fold, 'DT' : dt}
    
    # parameters
    epoch = 50
    pop_size = 10
    pp = 0.12
    A, epxilon = 2, 0.001
    lb = 0
    ub = 1

    ID_POS = 0
    ID_FIT = 1

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

    
    ######## calculate fitness of population
    for i in range(pop_size):
        sf_fit = np.append(sf_fit, Fun(xtrain, ytrain, sf_pop[i,:], opts))
    
    sf_fits = list(zip(sf_pop, sf_fit))
    
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
    PD = 1 - pop_size / (pop_size + s_size)
    AP = 1.5
    epxilon = 0.001


    for iterno in range(0, epoch):

        for idx in range(0, pop_size):
            lamda_i = 2 * np.random.uniform() * PD - PD
            pos_new = sf_gbest[ID_POS] - lamda_i * ( np.random.uniform() * ( sf_gbest[ID_POS] + s_gbest[ID_POS] ) / 2 - sf_fits[idx][ID_POS] )
            
            sf_pop_fit = sf_fits[idx][ID_FIT]
            new_tuple = (pos_new, sf_pop_fit)
            sf_fits[idx] = new_tuple

        AP = AP * (1 - 2 * (epoch + 1) * epxilon)
        print(AP)

        if AP < 0.5:
            for i in range(0, s_size):
                temp = (sf_gbest[ID_POS] + AP) / 2
                pos_new = np.random.uniform() * (temp - s_fits[i][ID_POS])
                s_pop_fit = s_fits[i][ID_FIT]
                new_tuple = ( pos_new, s_pop_fit)
                s_fits[i] = new_tuple
                print(new_tuple)
        else:
            for i in range(0, s_size):
                pos_new = np.random.uniform() * (sf_gbest[ID_POS] - s_fits[i][ID_POS] + AP)
                s_pop_fit = s_fits[i][ID_FIT]
                new_tuple = ( pos_new, s_pop_fit)
                s_fits[i] = new_tuple
        ## Recalculate the fitness of all sardine
        
        ############# position ha ra dar akhar binary konim
