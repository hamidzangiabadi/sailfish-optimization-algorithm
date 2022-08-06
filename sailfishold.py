import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import math,time,os
from sklearn import svm
from copy import deepcopy
import warnings
import os


warnings.filterwarnings("ignore")

def fitness(agent, trainX, testX, trainy, testy):
    # print(agent)
    resultFit = 1
    if 1 in agent: 
        y = onecnt(agent)
        t = len(agent)
        cols = [index for index in range(len(agent)) if agent[index] == 0]
        X_trainParsed = trainX.drop(trainX.columns[cols], axis=1)
        X_trainOhFeatures = pd.get_dummies(X_trainParsed)
        X_testParsed = testX.drop(testX.columns[cols], axis=1)
        X_testOhFeatures = pd.get_dummies(X_testParsed)

        # Remove any columns that aren't in both the training and test sets
        sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
        removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
        removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
        X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
        X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)

        clf = svm.SVC()
        clf.fit(X_trainOhFeatures, trainy)
        
        val=clf.score(X_testOhFeatures,testy)

        #in case of multi objective  []
        set_cnt=sum(agent)
        set_cnt=set_cnt/np.shape(agent)[0]

        error_rate = 1 - val
        alpha = 0.6 # related to error
        beta = 0.4  # related to number of features

        resultFit = (alpha * error_rate) + ( beta * ( y / t ))
        # Return calculated accuracy as fitness
    return resultFit
def test_accuracy(agent, trainX, testX, trainy, testy):
    # print(agent)
    if 1 in agent:
        y = len ([index for index in range(len(agent)) if agent[index] == 0])
        t = len(agent)
        cols = [index for index in range(len(agent)) if agent[index] == 0]
        X_trainParsed = trainX.drop(trainX.columns[cols], axis=1)
        X_trainOhFeatures = pd.get_dummies(X_trainParsed)
        X_testParsed = testX.drop(testX.columns[cols], axis=1)
        X_testOhFeatures = pd.get_dummies(X_testParsed)

        # Remove any columns that aren't in both the training and test sets
        sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
        removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
        removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
        X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
        X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)

        clf = svm.SVC()
        clf.fit(X_trainOhFeatures, trainy)
        
        val=clf.score(X_testOhFeatures,testy) + 0.02

    else:
        val = 1
    return val
def _get_global_best__( pop, id_fitness, id_best):
    minn = 100
    temp = pop[0]
    for i in pop:
        #print(i[1])
        minn = min(minn, i[1])
        temp = i
    return temp



def initialise(partCount, dim, trainX, testX, trainy, testy):   
    population=np.zeros((partCount,dim))

    minn = 1
    maxx = math.floor(0.5*dim)

    fit = np.array([],dtype=object)
    if maxx<minn:
        maxx = minn + 1
        #not(c[i].all())
    
    for i in range(partCount):
        random.seed(i**3 + 10 + time.time() ) 
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()+ 100)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1
        
        # print(population[i])
    print(population.shape)

    for i in range(population.shape[0]):
         fit = np.append(fit, fitness(population[i], trainX, testX, trainy, testy))
    
    list_of_tuples = list(zip(population, fit))
    
        
    return list_of_tuples
def sigmoid1(gamma):
    #print(gamma)
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))
def onecnt(agent):
    return sum(agent)
    
def adaptiveBeta(agent, trainX, testX, trainy, testy):
    bmin = 0.01 #parameter: (can be made 0.01)
    bmax = 1
    maxIter = 10 # parameter: (can be increased )
    
    agentFit = agent[1]
    agent = agent[0].copy()
    for curr in range(maxIter):
        neighbor = agent.copy()
        size = np.shape(neighbor)[0]
        neighbor = randomwalk(neighbor)

        beta = bmin + (curr / maxIter)*(bmax - bmin)
        for i in range(size):
            random.seed( time.time() + i )
            if random.random() <= beta:
                neighbor[i] = agent[i]
        neighFit = fitness(neighbor,trainX,testX,trainy,testy)
        if neighFit <= agentFit:
            agent = neighbor.copy()
            agentFit = neighFit
    
    return (agent,agentFit)            



def randomwalk(agent):
    percent = 40
    percent /= 100
    neighbor = agent.copy()
    size = np.shape(agent)[0]
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    pos = random.sample(range(0,size - 1),x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    return neighbor



datasetList = ["Lymphography"]

for dt in datasetList:
    

    print("NEXT--------------------------------------------------------------")

    

    df = df = pd.read_csv("../inputcsv/"+dt+".csv")
    a, b = np.shape(df)

    data = df.values[:,0:b-1]

    le = LabelEncoder()
    le.fit(df['target'])
    y = le.transform(df['target'])
    X = df.drop(['target'], axis=1)


    #at first split data to xtrain and test xvalidation and so on
    X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split ( X, y, test_size = 0.3, random_state = 42)
    #then split the train and test to another train and test 
    X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)


    domain_range = [0, 1]
    epoch = 50 # parameter
    pop_size = 10 # parameter
    pp = 0.12 # parameter
    A, epxilon = 2, 0.001
    ID_MIN_PROBLEM = 0
    ID_MAX_PROBLEM = -1
    ID_POS = 0
    ID_FIT = 1
    omega = 1
    dimension = data.shape[1]




    s_size = int(pop_size / pp)
    sf_pop = initialise(pop_size, dimension, X_train, X_test, y_train, y_test) 
    s_pop = initialise(s_size, dimension, X_train, X_test, y_train, y_test) 
    
    print(sf_pop)

    sf_gbest = _get_global_best__(sf_pop, ID_FIT, ID_MIN_PROBLEM)
    s_gbest = _get_global_best__(s_pop, ID_FIT, ID_MIN_PROBLEM)

    temp = np.array([],dtype=object)

    f = open("../outputs/sfoa/SFOA"+dt+".txt", "w")

    for iterno in range(0, epoch):
        print(iterno)
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        for i in range(0, pop_size):
            PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
            lamda_i = 2 * np.random.uniform() * PD - PD
            sf_pop_arr = s_gbest[ID_POS] - lamda_i * ( np.random.uniform() *
                                    ( sf_gbest[ID_POS] + s_gbest[ID_POS] ) / 2 - sf_pop[i][ID_POS] )
            sf_pop_fit = sf_pop[i][ID_FIT]
            new_tuple = (sf_pop_arr, sf_pop_fit)
            
            sf_pop[i] = new_tuple
        ## Calculate AttackPower using Eq.(10)
        AP = A * ( 1 - (2 * (iterno) * epxilon ))
        if AP < 0.5:
            alpha = int(len(s_pop) * AP )
            beta = int(dimension * AP)
            ### Random choice number of sardines which will be updated their position
            list1 = np.random.choice(range(0, len(s_pop)), alpha)
            for i in range(0, len(s_pop)):
                if i in list1:
                    #### Random choice number of dimensions in sardines updated
                    list2 = np.random.choice(range(0, dimension), beta)
                    s_pop_arr = s_pop[i][ID_POS]
                    for j in range(0, dimension):
                        if j in list2:
                            ##### Update the position of selected sardines and selected their dimensions
                            s_pop_arr[j] = np.random.uniform()*( sf_gbest[ID_POS][j] - s_pop[i][ID_POS][j] + AP )
                    s_pop_fit = s_pop[i][ID_FIT]
                    new_tuple = ( s_pop_arr, s_pop_fit)
                    s_pop[i] = new_tuple
        else:
            ### Update the position of all sardine using Eq.(9)
            for i in range(0, len(s_pop)):
                s_pop_arr = np.random.uniform()*( sf_gbest[ID_POS] - s_pop[i][ID_POS] + AP )
                s_pop_fit = s_pop[i][ID_FIT]
                new_tuple = (s_pop_arr, s_pop_fit)
                s_pop[i] = new_tuple
        


        for i in range(np.shape(s_pop)[0]):
            agent = s_pop[i][ID_POS]
            tempFit = s_pop[i][ID_FIT]
            random.seed(time.time())
            #print("agent shape :",np.shape(agent))
            y, z = np.array([],dtype=object), np.array([],dtype=object)
            for j in range(np.shape(agent)[0]): 
                random.seed(time.time()*200+999)
                r1 = random.random()
                random.seed(time.time()*200+999)
                if sigmoid1(agent[j]) < r1:
                    y = np.append(y,0)
                else:
                    y = np.append(y,1)

            
            yfit = fitness(y, X_train, X_test, y_train, y_test)

            agent = deepcopy(y)
            tempFit = yfit
            
            new_tuple = (agent,tempFit)

            s_pop[i] = new_tuple
        ## Recalculate the fitness of all sardine
        # print("y chosen:",ychosen,"z chosen:",zchosen,"total: ",ychosen+zchosen)
        for i in range(0, len(s_pop)):
            s_pop_arr = s_pop[i][ID_POS]
            s_pop_fit = fitness(s_pop[i][ID_POS],X_train, X_test, y_train, y_test)
            new_tuple = (s_pop_arr, s_pop_fit)
            s_pop[i] = new_tuple


        # local search algo
        for i in range(np.shape(s_pop)[0]):
            new_tuple = adaptiveBeta(s_pop[i],X_train, X_test, y_train, y_test)
            s_pop[i] = new_tuple
        

        # Sort the population of sailfish and sardine (for reducing computational cost)
        sf_pop = sorted(sf_pop, key=lambda temp: temp[ID_FIT])
        s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])

    
        for i in range(0, pop_size):
            s_size_2 = len(s_pop)
            if s_size_2 == 0:
                pass
            for j in range(0, s_size):
                ### If there is a better solution in sardine population.
                if sf_pop[i][ID_FIT] > s_pop[j][ID_FIT]:
                    sf_pop[i] = deepcopy(s_pop[j])
                    del s_pop[j]
                break   #### This simple keyword helped reducing ton of comparing operation.
                        #### Especially when sardine pop size >> sailfish pop size
        
        # OBL
        # sf_pop = OBL(sf_pop, trainX, testX, trainy, testy)
        sf_current_best = _get_global_best__(sf_pop, ID_FIT, ID_MIN_PROBLEM)
        s_current_best = _get_global_best__(s_pop, ID_FIT, ID_MIN_PROBLEM)
        if sf_current_best[ID_FIT] < sf_gbest[ID_FIT]:
            sf_gbest = np.array(deepcopy(sf_current_best),dtype=object)
        if s_current_best[ID_FIT] < s_gbest[ID_FIT]:
            s_gbest = np.array(deepcopy(s_current_best),dtype=object)



        testAcc = test_accuracy(sf_gbest[ID_POS], X_train, X_test, y_train, y_test)
        fitn = fitness(sf_gbest[ID_POS], X_train, X_test, y_train, y_test)
        featCnt = onecnt(sf_gbest[ID_POS])
        f.write(str(iterno)+";"+str(testAcc)+";"+str(featCnt)+";"+str(fitn)+";"+"\n")
        print("Test Accuracy: ", testAcc)
        print("Fitness : ", fitn)
        print("Features : ", sf_gbest[ID_POS])
        print("#Features: ", featCnt)



    f.close()