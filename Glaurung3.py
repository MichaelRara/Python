from __future__ import absolute_import, division, print_function
import tkinter as tk 
from tkinter import ttk
from tkinter import filedialog
from tkinter import * 
from tkinter.ttk import *
from tkinter import messagebox
import os as os
import pathlib
import _thread
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
#from sklearn.decomposition import PCA
import io
import random as rd
import math 
import sklearn
from sklearn.mixture import GaussianMixture 
from sklearn import datasets 
import seaborn as sns

def OwnLoss(y_true, y_pred):
    '''K.print_tensor(y_true, message='')
    K.set_epsilon(0.05)
    mape = K.abs( 100* ( y_true-y_pred) / y_true)    
    K.switch(K.less_equal(mape, K.epsilon()),K.update(mape, 0),K.pow(mape, (y_true-y_pred) * (y_true-y_pred)) )
    '''
    return K.mean( K.pow( K.abs(100*(y_true-y_pred)/y_true),2) )
    #return K.max(K.abs( 100* ( y_true-y_pred) / y_true ))
    #return K.mean(K.relu( K.pow( K.relu( K.abs( 100* ( y_true-y_pred) / y_true )),2 )-10))
    #return K.mean(K.pow(K.abs( 100* ( y_true-y_pred) / y_true),10)/ 1679616 )
    
    '''
    return K.mean(K.abs( 100* ( y_true-y_pred) / y_true) )
    mape = K.abs( 100* ( y_true-y_pred) / y_true)
    K.set_epsilon(0.05)
    K.flatten(mape)
    n=K.int_shape(mape)


    for i in range(0, n[0]):
        if K.less_equal(mape[i], K.epsilon()):
            K.update(mape[i], 0)
        else:
            K.pow(mape[i], (y_true-y_pred) * (y_true-y_pred))
    return K.mean(mape)
    '''
def Build():
    #CheckPoint()
    filename, dataset, H = LoadEntry()   
    '''
    if chkPCA.get():
        filename = 'PCA' + filename
    '''
    #CheckExtrapolation(dataset)
    dataset.iloc[:,1:], A = norm(dataset.iloc[:,1:].copy(),True)  
    NormMatrix = pd.DataFrame(A )
    #print(NormMatrix)
    # Aplikace PCA
    '''if chkPCA.get():
        X = dataset.iloc[:,1:].values
        print(X)
        m = dataset.shape[0]
        Sigma = 1/m * np.matmul(np.transpose(X),X)
        U, S, V = np.linalg.svd(Sigma)
        sumS = S.sum()
        lengthS = S.size
        s = 0
        for i in range (0,lengthS):
            s += S[i]
            if s/sumS >= 0.99:
                k=i
                break
        Ureduce = U[:,0:k+1]
        r,s = Ureduce.shape
        K = np.zeros((r,s))
        K=Ureduce
        Z = np.matmul( X,Ureduce)
        Y = dataset.iloc[:,0]
        dataset = pd.DataFrame(Z)
        dataset.insert(0, "Y==value",Y)
    '''
    if chkMeanMed.get():
        H1 = 'Prumer, Smerodatna odchylka'
    else: 
        H1 = 'Median, Smerodatna odchylka'
    # Rozdělení dat na části pro učení a testování
    train_dataset = dataset.sample(frac=float(txtTrainSet.get()),random_state=0) # parametr frac říká, jaká část nahraných dat bude použita pro učení a testování sítě. Zde tedy 0.8 značí, že 80% poskytnutých dat je použito pro učení a 20% pro testování. Zvykem je volit 0.7, ale dat nemáme mnoho.
    test_datasetOrigin = dataset.drop(train_dataset.index)
    # Oddělení funkčních hodnot od hodnot vstupních
    train_labels = train_dataset.iloc[:,0]
    test_labels = test_datasetOrigin.iloc[:,0]
    train_dataset = train_dataset.drop(train_dataset.columns[0],axis=1)
    test_datasetOrigin2 = normDataBack(test_datasetOrigin.iloc[:,1:].copy(),NormMatrix)
    test_datasetOrigin2.insert(0,'Function values',test_labels.copy() )
    test_dataset = test_datasetOrigin.drop(test_datasetOrigin.columns[0],axis=1)
    #test_datasetOrigin2.insert(0,'Function values',test_labels.copy() )
    
    # Tvorba a učení jednotlivých neuronových sítí
    MinLayers = int(txtHLOd.get())
    MaxLayers = int(txtHLDo.get())
    MinNeurons = int(txtNOd.get())
    MaxNeurons = int(txtNDo.get())
    r = int(txtRestart.get())
    # Vytvoření složky pro učení neuronové sítě
    txtName = 'ZaznamUceniRegrese'
    ComboLossFc = comboLossFc.get()
    if ComboLossFc=='MAE': 
        LossFc = 'mean_absolute_error'
    elif ComboLossFc=='MSE':
        LossFc = 'mean_squared_error'
    elif ComboLossFc=='MAPE':
        LossFc = 'mean_absolute_percentage_error'
    #chyba = LossFc
    '''
    T = chkRegrese_Klasifikace.get()
    if T: 
        txtName = 'ZaznamUceniRegrese'
        chyba = 'Chyba = mae'
    else:
        txtName = 'ZaznamUceniKlasifikace'
        chyba = 'Přesnost'
    '''
    txtFinalName = './' +txtName+ str(MinLayers)+'_'+str(MaxLayers)+'-'+str(MinNeurons)+'_'+str(MaxNeurons)+'-'+str(r)+'.txt'
    file1 = open(txtFinalName,"a") 
    #file1.write('Vrstev, Neuronu, '+chyba +'\n')
    file1.write('Hidden layers, Neurons, Mean Absolute Error, Mean Squared Error, Mean Absolute Percentage Error ' +'\n')
    # Samotná interpolace dat
    for L in range(MinLayers, MaxLayers+1):
        for N in range(MinNeurons,MaxNeurons+1):
            file1 = open(txtFinalName,"a") 
            for i in range(0,r):
                # Sestavení architektury neuronové sítě
                if L>0 and N==0:
                    continue
                else:    
                    model = build_model(train_dataset,L,N,LossFc)
                # Odchylka
                history = Learn(model,train_dataset, train_labels)
                model.load_weights("training_1/cp.ckpt")
                loss, mae, mse,mpe = model.evaluate(test_dataset, test_labels, verbose=0)
                
                '''
                if T:
                    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
                else:
                    results = model.evaluate(test_dataset, test_labels, verbose=0)
                    mae = results[1]
                '''
                # Vytvoření složky
                Name=filename+'Layers'+str(L)+'Neurons'+str(N)+'Epochs'+txt7.get()+ComboLossFc+str(round(loss,3))
                os.makedirs('./'+Name)
                # Uložení grafů učení
                plot_history(history,Name)
                '''
                if chkPCA.get():
                    # Uložení redukční matice Ureduce do txt
                    np.savetxt('./' + Name + '/' + 'Ureduce.txt',K , delimiter = ',', newline = '\n', comments = ' ')
                    # Uložení zredukovaných dat do txt
                    np.savetxt('./' + Name + '/' + 'ReduceData.txt',dataset , delimiter = ',', newline = '\n', comments = ' ')
                '''
                # Uložení testovací množiny a jejich funkčních hodnot
                np.savetxt('./' + Name + '/' + 'TestDataset.txt',test_datasetOrigin2 , delimiter = ',', newline = '\n', header = H, comments = ' ')
                # Uložení normalizační trenovaci matice a normalizacni testovaci matice
                np.savetxt('./' + Name + '/' + 'NormalizationMatrix.txt',A,delimiter=',', newline='\n',header=H1,comments='')
                #np.savetxt('./' + Name + '/' + 'NormalizacniTestovaciMatice.txt',B,delimiter=',', newline='\n',header=H1,comments='')
                # Uložení vah a modelu
                model.save_weights('./'+Name+'/'+'Weights') # Uloží váhy modelu
                model.save('./'+Name+'/'+Name+'.h5') # Save entire model to a HDF5 file
                # Uložení matice vah do txt souboru
                with open('./' + Name + '/' +'Matice vah.txt', 'w') as f:
                    f.write(str(model.layers[0].get_weights()))
                # Regrese a uložení predikovaných hodnot
                test_predictions = model.predict(test_dataset).flatten()
                np.savetxt('./' + Name + '/' + 'TestDatasetResults.txt',test_predictions, delimiter=',', newline='\n',header="Predicted values",comments='')
                Regrese(test_labels, test_predictions,'./'+Name)
                # Výpočet odchylky a analýza odchylky (střední hodnota rezidua, směrodatná odchylka)
                E, O = Rezidua(test_predictions, test_labels)
                if i==0:
                    minn = loss
                elif mae < minn:
                    minn = loss
                Info(E,O,L,N,mae,mse,mpe,H,'./' + Name + '/' + 'Info.txt')
            file1.write(str(L)+', '+ str(N)+', '+str(mae)+', '+str(mse)+', '+str(mpe)+ '\n') # Zapsání údajů do txt souboru
            file1.close() # Zavření txt souboru
    messagebox.showinfo('Glaurung','Algortihms has finished succesfully!')
def build_model(train_dataset,m,n,LossFc): # m = počet skrytých vrstev, n = počet neuronů v každé vrstvě
    model = keras.Sequential()
    NmbOfFeatures=len(train_dataset.keys())
    a=0.01 #0.01 parametr pro LeakyRelu funkci
    # T = chkRegrese_Klasifikace.get()
    if m==0:
        model.add(keras.layers.Dense(1, input_shape=[NmbOfFeatures]))
    else:
        model.add(keras.layers.Dense(n, input_shape=[NmbOfFeatures]))
        for i in range(0,m-1): # Vytvoří skryté vrstvy
            #model.add(keras.layers.Dense(n, activation='relu'))
            model.add(keras.layers.Dense(n))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=a))
           # model.add(BatchNormalization())
           # model.add(Dropout(0.5))
        # model.add(keras.layers.Dense(1, activation='relu'))
        # if T:
        model.add(keras.layers.Dense(1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=a))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # else:
        # model.add(keras.layers.Dense(1), activation=tf.nn.sigmoid )
    optimizer = tf.keras.optimizers.Adam(float(txtStep.get()))
    #model.compile(loss=LossFc, optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error','mean_absolute_percentage_error'])
    model.compile(loss=LossFc, optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error','mean_absolute_percentage_error'])
    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error','mean_absolute_percentage_error'])
    '''
    if T:
        optimizer = tf.keras.optimizers.Adam(float(txtStep.get()))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
    else:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    '''
    return model
def CheckExtrapolation(x,CovMatrix,meanC,WantedProbmin,WantedProbmax):
    px,pmean = Probability(x,meanC,CovMatrix)
    r = x.shape[0]
    NicePoints = []
    # Ověření, jestli bod leží ve vnitřní oblasti daného clusteru
    # Parametr 0.25 je testovací kritérium, čím menší, tím větší přípustná oblast clusteru bude
    for i in range(0,r):
        if px[i] >= WantedProbmin*pmean and px[i]<=pmean*WantedProbmax :
        #if px[i] >= 0.25*pmean:
            NicePoints.append(x.iloc[i,:])
    NicePoints = pd.DataFrame(NicePoints)
    return NicePoints
def CheckPoints(x, PathToClusterDir,WantedProbmin,WantedProbmax):
    i = 0
    UsefullPoints = pd.DataFrame()
    means = pd.read_csv(PathToClusterDir + '/Means.txt', na_values = "?",header=None ,comment='\t', sep = ",", skipinitialspace=True)
    while os.path.exists(PathToClusterDir + '/CovarianceMatrix' + str(i) + '.txt'):
        CovMatrix = pd.read_csv(PathToClusterDir + '/CovarianceMatrix' + str(i) + '.txt', na_values = "?",header=None, comment='\t', sep = ",", skipinitialspace=True)
        UsefullPoints = CheckExtrapolation(x,CovMatrix,means.iloc[i,:],WantedProbmin,WantedProbmax )
        if len(UsefullPoints.index) != 0:
            return True, UsefullPoints;
        i += 1
    return False, UsefullPoints;
def CheckPointsInCluster(x,PathToClusterDir, meanC, CovMatrix,WantedProbmin,WantedProbmax):
    UsefullPoints = pd.DataFrame()
    UsefullPoints = CheckExtrapolation(x,CovMatrix,meanC,WantedProbmin,WantedProbmax)
    if len(UsefullPoints.index) != 0:
        return True, UsefullPoints;
    return False, UsefullPoints;
def Clustering():
    dataset = LoadData(False)
    X = dataset.iloc[:,1:].copy() 
    #X = dataset.copy() 
    #X= norm(X,False)
    d = pd.DataFrame(X)
    n_components = int(txtMaxClusters.get())
    A = np.zeros((n_components,2))
    for n in range(1,n_components+1):
        print('Cluster'+str(n))
        for i in range(0,20):
            clfs = GaussianMixture(n, max_iter = 1000).fit(X) 
            aics = clfs.aic(X)
            if i==0:
                minAics = aics
            elif minAics > aics:
                minAics = aics
        A[n-1,0] = n
        A[n-1,1] = minAics
    plt.plot(A[:,0],A[:,1])
    plt.figure()
    plt.xlabel('Count of clusters')
    plt.ylabel('Error')
    plt.plot(A[:,0],A[:,1])
    plt.legend()
    plt.savefig('./ClusteringPicture.png', bbox_inches='tight')
def ClusterInfo(data,PathToClusterDir ):
    means = pd.read_csv(PathToClusterDir + '/Means.txt', na_values = "?",header=None ,comment='\t', sep = ",", skipinitialspace=True)
    i = 0
    maxVector = np.zeros((data.shape[0],3))
    while os.path.exists(PathToClusterDir + '/CovarianceMatrix' + str(i) + '.txt'):
        CovMatrix = pd.read_csv(PathToClusterDir + '/CovarianceMatrix' + str(i) + '.txt', na_values = "?",header=None, comment='\t', sep = ",", skipinitialspace=True)
        Prob = Probability(data, means.iloc[i,:], CovMatrix)
        for j in range(0, data.shape[0]):
            if Prob[0][j] >= maxVector[j,0]:
                maxVector[j,0] = Prob[0][j]
                maxVector[j,1] = Prob[0][j] / Prob[1]
                maxVector[j,2] = i
        i += 1
    return maxVector
def CheckPath(NormMat,OldState,NewState,WantedVals,model,meanC,CovMatrix,WantedProb):
    # r,s = OldState.shape
    pocetUzlu = int(txtUzly.get())
    w = pocetUzlu-1
    delkaKroku = 1/w
    v = NewState[3:] - OldState[2:] # vektor posunutí
    i = 1
    t = delkaKroku
    State2 = np.zeros((1,len(v)))
    while t < 1:
        State2 =  OldState[2:] + t * v
        '''for j in range(0, len(v)):
            State2[0,j] = OldState[j+2] + t * v[j]
        '''
        # State = np.reshape(State, (1, len(State)))
        # State = pd.DataFrame( [[OldState[2:] + t * v]] )
        State = pd.DataFrame([State2])
        # Normování dat
        State = normData(State.copy(),NormMat)
        fceValue = model.predict(State).flatten()
        # Zpětná transformace znormovaných dat
        State = normDataBack(State, NormMat)
        prob, probMean = Probability(State,meanC,CovMatrix)
        if fceValue > WantedVals[1] or fceValue < WantedVals[0] or prob < WantedProb[0]*probMean or prob > WantedProb[1]*probMean:
            return False 
        i += 1 
        t = i * delkaKroku
    return True
def CreateRecept(Path,NormMat,model,meanC, CovMatrix):
    r = len(Path)
    s = len(Path[0])
    pocetUzlu = int(txtUzly.get())
    w = pocetUzlu-1
    delkaKroku = 1/w
    State = []
    E = 1 - delkaKroku
    t = np.arange(0,1,delkaKroku).reshape(w,1)
    Recept = pd.DataFrame()
    for i in range(0,r-1):
        A = np.array(Path[i+1])
        B = np.array(Path[i]) 
        v = A - B # vektor posunutí
        v = np.reshape(v, (1, s) )
        B2 = np.zeros((w,1),dtype=B.dtype) + B
        C = B2 + t * v
        PartialRecept = pd.DataFrame(C)
        Recept = Recept.append(PartialRecept,ignore_index=True)
    B = np.array(Path[r-1])
    State.append(B)
    PartialRecept = pd.DataFrame(State)
    Recept = Recept.append(PartialRecept,ignore_index=True)
    # Odstranění sloupců rodičů 
    #Recept = Recept.drop(Recept.columns[1], axis=1)
    Recept = Recept.drop(Recept.columns[0], axis=1)
    # Znormování vstupních hodnot
    Recept = normData(Recept,NormMat)
    # Napočítání funkčních hodnot
    Results = model.predict(Recept).flatten()
    # Zpětná transformace znormovaných dat
    Recept = normDataBack(Recept,NormMat)
    # Napočítání pravděpodobností
    prob, probMean = Probability(Recept,meanC, CovMatrix)
    Recept.insert(0,'Function values', Results)
    Recept.insert(0,'DistValue/DistValueMean', prob/probMean)
    return Recept
def CreateFirstGeneration(NmbOfChrom, s, StoneParameters, CovMatrix, meanC, alpha, MinMax): # Vytvoří prvotní generaci - generovaná náhodně
    Generation = pd.DataFrame(np.zeros((NmbOfChrom,s)))
    mean = np.round(meanC, 0)
    diagonal = np.sqrt(np.diagonal(CovMatrix))
    variances = np.round(alpha * diagonal, 0) #směrodatná odchylka
    variance2 =  alpha * diagonal
    for i in range (0,NmbOfChrom):
        for j in range (0, s):
            if (j in StoneParameters)==False:
                if MinMax[2,j] == 1:
                    Generation.iloc[i,j] = rd.randint( mean[j] - variances[j], mean[j] + variances[j] )
                    #Generation.iloc[i,j] = rd.randint(MinMax[0,j],MinMax[1,j])
                elif MinMax[2,j] == 0:
                    line = rd.randint(0,1)
                    Generation.iloc[i,j] = MinMax[line,j]
                else:
                    #Sequence = list(np.arange( meanC[j] - variance2[j], meanC[j] + variance2[j], MinMax[2,j]))# MinMax[0,j],MinMax[1,j],MinMax[2,j]))
                    Sequence = list(np.arange( meanC[j] - variance2[j], meanC[j] + variance2[j], variance2[j]*0.01 ))
                    Generation.iloc[i,j] = rd.choice(Sequence)
                    #Sequence = list(np.arange(MinMax[0,j],MinMax[1,j],MinMax[2,j]))
                    #Generation.iloc[i,j] = rd.choice(Sequence)
            else:
                Generation.iloc[i,j] = StoneParameters[j]
    return Generation 
def Evolution(model,Generation,WantedValue,distance,StoneParameters,MinMax,NormMat,alpha,meanC=None,CovMatrix=None, T=True,fitness='values'):
    It =  int(txtFindIter.get()) # Počet generací na vytvoření
    PerfectGeneration = []
    for k in range (0,It):
        print('Generace ' + str(k))
        Generation = Generation_Fitness(model, Generation.copy(), WantedValue,NormMat, meanC, CovMatrix, S = fitness)
        # print(Generation)
        # Pokud je hodnota fitness funkce menší rovna distance, pak se jedinec uloží do PerfectGeneration
        numOfRows = Generation.shape[0] # Zjistí počet řádků
        for i in range(0,numOfRows):
            if Generation.iloc[i,0] <= distance:
                PerfectGeneration.append(Generation.iloc[i,1:].copy())
                if T == False:
                    return Generation.iloc[i,1:].tolist(), False;
            else:
                break
        # Aplikování křížení, mutace a zrození/smrt na původní generaci 
        Generation = EvolvedGeneration(Generation.copy(),MinMax,NormMat,StoneParameters,CovMatrix, meanC, alpha)
    if T==False:
            print('Algorithm was not able to find start point. I recommend you to change entry parameters.')
            exit()
    return Generation, PerfectGeneration;
def Evolution2(model,It,Generation,Tolerance,StoneParameters,MinMax,NormMat,alpha,meanC,CovMatrix,fitness, T=True):
    PerfectGeneration = []
    for k in range (0,It):
        print('Generace ' + str(k))
        Generation = Generation_Fitness2(model, Generation.copy(), Tolerance, NormMat, meanC, CovMatrix, fitness)
        # print(Generation)
        # Pokud je hodnota fitness funkce menší rovna distance, pak se jedinec uloží do PerfectGeneration
        numOfRows = Generation.shape[0] # Zjistí počet řádků
        for i in range(0,numOfRows):
            if Generation.iloc[i,0] <= 1:
                PerfectGeneration.append(Generation.iloc[i,1:].copy())
                if T == False:
                    return Generation.iloc[i,1:].tolist(), False;
            else:
                break
        # Aplikování křížení, mutace a zrození/smrt na původní generaci 
        Generation = EvolvedGeneration(Generation.copy(),MinMax,NormMat,StoneParameters,CovMatrix, meanC, alpha)
    if T==False:
            print('Algorithm was not able to find start point. I recommend you to change entry parameters.')
            exit()
    return Generation, PerfectGeneration;
def EvolvedGeneration(Generation,MinMax,NormMat,StoneParameters,CovMatrix, meanC, alpha): # Vrátí neznormovanou novou generaci
    # Smazání prvního (Function values) a nultého (Fitness) sloupce  
    Generation = Generation.drop(Generation.columns[1], axis=1)
    Generation = Generation.drop(Generation.columns[0], axis=1)
    r,s = Generation.shape
    Generation2 = pd.DataFrame(np.zeros((r,s)))
    Generation2 = Generation.copy()
    # Zkopírování nejlepšího jedince do Generation2 
    Generation2.iloc[0,:] = Generation.iloc[0, :].copy()
    r1 = r // 2
    r2 = r1 // 2 + r1 
    # 1/2 generace je použita pro křížení
    for i in range(1,r1):
        T = False
        # Nalezení partnera pro křížení
        while T == False:
            j = rd.randint(0,r1-1)
            if j != i:
                T = True
        Parent1 = Generation.iloc[i, :].copy()
        Parent2 = Generation.iloc[j, :].copy()
        Child = Parent1.copy()
        #s2 = len(Parent1) 
        # Křížení rodičů
        #for j in range(0,s2):
        for j in range(0,s):
            if rd.randint(1,2) == 2: # Pokud se vygeneruje číslo 2, pak dítě zdědí vlastnost 2. rodiče
                Child.iloc[j] = Parent2.iloc[j]
        # Nahrání potomka do nové generace
        Generation2.iloc[i] = Child.copy() 
    # 1/4 generace je použita pro mutaci
    for i in range(r1,r2):
        for j in range(0,s):
            a = rd.randint(0,1)
            if (a == 1) and ((j in StoneParameters)==False): # Zahaj mutaci j-tého parametru
                Generation2.iloc[i,j] = Mutation(j,MinMax,NormMat,CovMatrix, meanC, alpha)
    # 1/4 generace je použita pro vygenerování nových členů
    for i in range(r2,r):
        for j in range(0,s):
            if (j in StoneParameters)==False:
                Generation2.iloc[i,j] = Mutation(j,MinMax,NormMat,CovMatrix, meanC, alpha)
            #else:
                #Generation2.iloc[i,j] = (StoneParameters[j] - NormMat.iloc[j,0] ) / NormMat.iloc[j,1]
                #Generation2.iloc[i,j] = StoneParameters[j]
    # Překopírování nové generace do Generace 
    Generation = Generation2.copy()
    return Generation
def FindParameters():
    # Načtení základních vstupních dat
    # filename,dataset, H,model,Dir,NormMat,MinMax,s = LoadAll()
    # Načtení dat
    filename, data, H = LoadEntry()
    # Zkopírování hodnot z csv souboru do pole MinMax
    r,s = data.shape
    r2 = 5
    MinMax = np.zeros((r2,s)) 
    for i in range (0,r):
        MinMax[i]=data.loc[i]
    # Načtení složky
    PathToModel = filedialog.askdirectory(title = 'Vyber složku obsahující model')
    # Dir = PathToModel
    # Načtení modelu/ů
    DirName = os.path.basename(PathToModel)
    # Načtení neuronové sítě
    model = keras.models.load_model(PathToModel +'/'+DirName+'.h5')
    # Načítání normalizační matice a normování dat
    NormMat = pd.read_csv(PathToModel + '/NormalizacniMatice.txt', sep = ",", header = 0)   
    NormMat = pd.DataFrame(NormMat) 
        
    NmbOfChrom = int(txtFindJed.get()) # Počet jedinců v jedné generaci
    WantedValuemin = float(txtFindFcemin.get())
    WantedValuemax = float(txtFindFcemax.get())
    WantedValueMean = (WantedValuemax+WantedValuemin)/2
    distance = (WantedValuemax-WantedValuemin)/2
    # Načtení požadovaných pravděpodobností
    WantedProbmin = float(txtProbmin.get())
    WantedProbmax = float(txtProbmax.get())
    WantedProbMean = (WantedProbmax+WantedProbmin)/2
    #distanceProb = (WantedProbmax-WantedProbmin)/2
    # Indexi clusterů
    ClusterMin = int(txtClusterMin.get())
    ClusterMax = int(txtClusterMax.get()) + 1 
    # Načtení pevných vstupních parametrů 
    StoneParameters = LoadStoneParameters(s,MinMax)
    # Načtení složky obsahující střední hodnoty a covariační matice clusterů
    PathToClusterDir = filedialog.askdirectory(title = 'Vyber složku obsahující clustery')
    # Vytvoření nulové generace
    # Generation = pd.DataFrame(np.zeros((NmbOfChrom, s)))
    # Zaplnění vytvořené generace
    It =  int(txtFindIter.get()) # Počet generací na vytvoření
    Tolerance = np.zeros((2,2))
    Tolerance[1,0] = WantedValueMean
    Tolerance[1,1] = distance
    r = NmbOfChrom
    r2 = r * 10
    alpha = -4 * WantedProbmin + 4 
    means = pd.read_csv(PathToClusterDir + '/Means.txt', na_values = "?",header=None ,comment='\t', sep = ",", skipinitialspace=True)
    #Generation = CreateFirstGeneration(r,s,StoneParameters,MinMax)
    for ClusterIndex in range(ClusterMin, ClusterMax):
        Generation = pd.DataFrame()
        Generation2 = pd.DataFrame()
        Generation3 = pd.DataFrame()
        #UsefullGeneration = pd.DataFrame()
        GenerationOk = pd.DataFrame()
        T = False
        meanC = means.iloc[ClusterIndex,:]
        CovMatrix = pd.read_csv(PathToClusterDir + '/CovarianceMatrix' + str(ClusterIndex) + '.txt', na_values = "?",header=None, comment='\t', sep = ",", skipinitialspace=True)
        pmean = Probability(GenerationOk,meanC,CovMatrix) 
        DistanceDistributionValue = pmean*(WantedProbmax-WantedProbmin)/2
        WantedDistributionValue = WantedProbMean*pmean
        Tolerance[0,0] = WantedDistributionValue
        Tolerance[0,1] = DistanceDistributionValue
        # Iteruje dokud nenalezne množinu bodů (velikosti alespoň r2) splňujících pravděpodobnostní podmínku
        #while T == False:
        Generation2 = CreateFirstGeneration(r, s, StoneParameters, CovMatrix, meanC, alpha, MinMax)
        Generation2, PerfectGeneration = Evolution2(model, It, Generation2.copy(),Tolerance, StoneParameters,MinMax,NormMat,alpha,meanC,CovMatrix,'ProbValue',True)
        maxVector = ClusterInfo(Generation2,PathToClusterDir )
        Generation2.insert(0,'Cluster',maxVector[:,2])
        Generation2 = Generation2[Generation2['Cluster'] == ClusterIndex ]
        Generation2 = Generation2.drop(Generation2.columns[0],axis=1)
        if Generation2.shape[0] > 0:
            E, Generation3 = CheckPointsInCluster(Generation2, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
            #Generation2 = CreateFirstGeneration(r2, s, StoneParameters, CovMatrix, meanC, alpha, MinMax)
            #Generation2, PerfectGeneration = Evolution(model, Generation2.copy(), WantedDistributionValue,DistanceDistributionValue,StoneParameters,MinMax,NormMat,alpha,meanC,CovMatrix,True,'probability')
            # print(Generation2)
            # print(PerfectGeneration)
            #E, Generation3 =  CheckPointsInCluster(Generation2, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
            # print(Generation3)
            if E:
                # Normování dat
                Generation3 = normData(Generation3.copy(),NormMat)
                # Predikce funkčních hodnot k získaným datům pomocí genetického algoritmu
                Results = model.predict(Generation3).flatten()
                # Zpětná transformace znormovaných dat
                Generation3 = normDataBack(Generation3,NormMat)
                # Nahrání funkčních hodnot do prvního sloupce
                Generation3.insert(0,'Function values', Results)
                Generation3 = Generation3[(Generation3['Function values'] >= WantedValuemin ) & (Generation3['Function values'] <= WantedValuemax )].copy()
            FinalPerfectGeneration = pd.DataFrame(PerfectGeneration)
            Generation = Generation3.append(FinalPerfectGeneration.copy(),ignore_index=True)
        elif len(PerfectGeneration)>0:
            Generation = pd.DataFrame(PerfectGeneration)
        # Napočítání pravděpodobností
        if Generation.shape[0] > 0:
            prob, pmean = Probability(Generation.iloc[:,1:],meanC,CovMatrix)
            Generation.insert(0,'ProbabilityX/probMean', prob/pmean)
            Generation.insert(0,'ProbabilityX', prob)
            # Uložení dat
            np.savetxt(PathToModel + '/' + 'Parametry' + 'Cluster' + str(ClusterIndex) +'.txt', Generation, delimiter=',', newline='\n', header = 'Probability, ProbabilityX/probMean, Function values,'+H, comments = '')
        '''
            #for i in range(0,s):
            #    FinalPerfectGeneration.iloc[:,i+1] = FinalPerfectGeneration.iloc[:,i+1] * NormMat.iloc[i,1] + NormMat.iloc[i,0]
            # Generation = Generation3.append(FinalPerfectGeneration.copy(),ignore_index=True)
            if Generation.shape[0] >= r2:
                # Uložení již vyhovujících bodů 
                # selecting rows based on condition 
                GenerationOk = Generation[(Generation['Function values'] >= WantedValuemin) & (Generation['Function values'] <= WantedValuemax)].copy()
                print('Byla nalezena množina bodů vyhovujícím pravděpodobnostním omezením. Nyní pracuji na nalezení požadovaných funkčních hodnot.')
                T = True
        # Nalezli jsme množinu bodů, které splňují pravděpodobnostní podmínku, nyní musíme splnit podmínku funkčních hodnot    
        # Normování dat
        # Generation = normData(Generation.iloc[:,1:].copy(),NormMat)
        # Aplikování genetického algoritmu
        Generation,PerfectGeneration2 = Evolution(model,Generation.iloc[:,1:].copy(),WantedValueMean,distance,StoneParameters,MinMax,NormMat,alpha,meanC,CovMatrix,True,'values')
        # Zpětná transformace znormovaných dat
        #Generation = normDataBack(Generation, NormMat)
        # Ověření podmínky pravděpodobnosti
        E, Generation = CheckPointsInCluster(Generation, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
        if E:
            # Normování dat
            Generation = normData(Generation.copy(),NormMat)
            # Predikce funkčních hodnot k získaným datům pomocí genetického algoritmu
            Results = model.predict(Generation).flatten()   
            # Zpětná transformace znormovaných dat
            Generation = normDataBack(Generation,NormMat)
            # Nahrání funkčních hodnot do prvního sloupce
            Generation.insert(0,'Function values', Results)
            # Ponecháme pouze požadované hodnoty
            Generation = Generation[(Generation['Function values'] >= WantedValuemin) & (Generation['Function values'] <= WantedValuemax)]
        
        if len(PerfectGeneration2) > 0:
            FinalPerfectGeneration = pd.DataFrame(PerfectGeneration2)
            # Odstranění sloupce funkčních hodnot
            FinalPerfectGeneration = FinalPerfectGeneration.drop(FinalPerfectGeneration.columns[0], axis=1)
            # Ověření podmínky pravděpodobnosti
            E, FinalPerfectGeneration = CheckPointsInCluster(FinalPerfectGeneration, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
            if E:
                FinalPerfectGeneration = normData(FinalPerfectGeneration.copy(),NormMat)
                # Predikce funkčních hodnot k získaným datům pomocí genetického algoritmu
                Results = model.predict(FinalPerfectGeneration).flatten()   
                # Zpětná transformace znormovaných dat
                FinalPerfectGeneration = normDataBack(FinalPerfectGeneration,NormMat)
                # Nahrání funkčních hodnot do prvního sloupce
                FinalPerfectGeneration.insert(0,'Function values', Results)
                # Ponecháme pouze požadované hodnoty
                FinalPerfectGeneration = FinalPerfectGeneration[(FinalPerfectGeneration['Function values'] >= WantedValuemin) & (FinalPerfectGeneration['Function values'] <= WantedValuemax)]
                Generation = Generation.append(FinalPerfectGeneration, ignore_index=True)
        if GenerationOk.shape[0] > 0:
            Generation = Generation.append(GenerationOk, ignore_index=True)
        if Generation.shape[0] > 0:
            Generation = Generation.sort_values('Function values')
            # Napočítání pravděpodobností     
            prob, pmean = Probability(Generation.iloc[:,1:],meanC,CovMatrix)
            Generation.insert(0,'ProbabilityX/probMean', prob/pmean)
            Generation.insert(0,'ProbabilityX', prob)
            # Uložení dat
            np.savetxt(Dir + '/' + 'Parametry' + 'Cluster' + str(ClusterIndex) +'.txt', Generation, delimiter=',', newline='\n', header = 'Probability, ProbabilityX/probMean, Function values,'+H, comments = '')
        '''
    messagebox.showinfo('Glaurung','Výpočet parametrů proběhl úspěšně!')
def FindRecept():
    # Načtení základních dat
    # filename,dataset, H,model,Dir,NormMat,MinMax,s = LoadAll()
    # Načtení dat
    filename, dataset, H = LoadEntry()
    # Zkopírování hodnot z csv souboru do pole MinMax
    r,s = dataset.shape
    r2 = 5
    MinMax = np.zeros((r2,s)) 
    for i in range (0,r):
        MinMax[i]=dataset.loc[i]
    # Načtení složky
    PathToModel = filedialog.askdirectory(title='Vyber složku obsahující model')
    # Dir = PathToModel
    # Načtení modelu/ů
    DirName = os.path.basename(PathToModel)
    # Načtení neuronové sítě
    model = keras.models.load_model(PathToModel +'/'+DirName+'.h5')
    # Načítání normalizační matice a normování dat
    NormMat = pd.read_csv(PathToModel + '/NormalizacniMatice.txt', sep = ",", header = 0)   
    NormMat = pd.DataFrame(NormMat) 
    
    
    
    r,s1 = dataset.shape
    r2 = 10 * int(txtFindJed.get())
    # Vytvoření listu Open a slovníku Closed
    Open = []
    Closed = {}
    # Načtení základních vstupů z okna aplikace
    NmbOfChrom = int(txtFindJed.get())  # Počet jedinců v jedné generaci
    WantedValuemin = float(txtFindFcemin.get())
    WantedValuemax = float(txtFindFcemax.get())
    WantedValueMean = (WantedValuemax+WantedValuemin)/2
    distance = (WantedValuemax-WantedValuemin)/2 # Maximální možná vzdálenost od středu intervalu hledané hodnoty
    PointDistance = float(txtFindRecept.get()) # Maximální možná vzdálenost dvou bodů
    It =  int(txtFindIter.get()) # Počet generací na vytvoření
    # Načtení pevných vstupních parametrů 
    # Načtení požadovaných pravděpodobností
    WantedProbmin = float(txtProbmin.get())
    WantedProbmax = float(txtProbmax.get())
    WantedProbMean = (WantedProbmax+WantedProbmin)/2
    # Indexi clusterů
    ClusterMin = int(txtClusterMin.get())
    ClusterMax = int(txtClusterMax.get()) +1 
    StoneParameters = LoadStoneParameters(s,MinMax)
    # Načtení složky obsahující střední hodnoty a covariační matice clusterů
    PathToClusterDir = filedialog.askdirectory('Vyber složku obsahující clustery')
    # Startovací pozice
    position = MinMax[0,0]
    # Vytvoření nulové generace
    # Hledání startovacího bodu
    alpha = -4 * WantedProbmin + 4 
    means = pd.read_csv(PathToClusterDir + '/Means.txt', na_values = "?",header=None ,comment='\t', sep = ",", skipinitialspace=True)
    for ClusterIndex in range(ClusterMin,ClusterMax):
        Generation = pd.DataFrame()
        meanC = means.iloc[ClusterIndex,:]
        CovMatrix = pd.read_csv(PathToClusterDir + '/CovarianceMatrix' + str(ClusterIndex) + '.txt', na_values = "?",header=None, comment='\t', sep = ",", skipinitialspace=True)
        pmean = Probability(Generation,meanC,CovMatrix)
        Generation2 = pd.DataFrame()
        Generation3 = pd.DataFrame()
        GenerationOk = pd.DataFrame()
        DistanceDistributionValue = pmean * (WantedProbmax-WantedProbmin)/2
        WantedDistributionValue = WantedProbMean * pmean
        '''while T:
            print('Hledám startovací bod')
            # Zaplnění vytvořené generace
            Generation = CreateFirstGeneration(NmbOfChrom,s,StoneParameters,CovMatrix, meanC, alpha,MinMax)
            # Normování dat
            #Generation = normData(Generation.copy(),NormMat)
            # Nalezení startovacího bodu
            #start, T = Evolution(model,Generation.copy(),WantedValueMean,distance,StoneParameters,MinMax,NormMat,False,fitness='total')
            #if T == False:
            #    break
            start, startPerfect = Evolution(model,Generation.copy(),WantedValueMean,distance,StoneParameters,MinMax,NormMat,True,fitness='value')
            res = model.predict(start).flatten()
            startPerfect = pd.DataFrame(startPerfect)
        '''
        # Iteruje dokud nenalezne množinu bodů (velikosti alespoň r2) splňujících pravděpodobnostní podmínku
        T=False
        while T == False:
            print('I am finding start point for cluster '+ str(ClusterIndex))
            Generation2 = CreateFirstGeneration(r2, s1, StoneParameters, CovMatrix, meanC, alpha, MinMax)
            Generation2, PerfectGeneration = Evolution(model, Generation2.copy(), WantedDistributionValue,DistanceDistributionValue,StoneParameters,MinMax,NormMat,alpha,meanC,CovMatrix,True,'probability')
            # Ověří, jestli bod leží v clusteru
            maxVector = ClusterInfo(Generation2,PathToClusterDir )
            Generation2.insert(0,'Cluster',maxVector[:,2])
            Generation2 = Generation2[Generation2['Cluster'] == ClusterIndex ]
            Generation2 = Generation2.drop(Generation2.columns[0],axis=1)           
           
           # print(Generation2)
           # print(PerfectGeneration)
            E, Generation3 =  CheckPointsInCluster(Generation2, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
           # print(Generation3)
            if E:
                # Normování dat
                Generation3 = normData(Generation3.copy(),NormMat)
                # Predikce funkčních hodnot k získaným datům pomocí genetického algoritmu
                Results = model.predict(Generation3).flatten()
                # Zpětná transformace znormovaných dat
                Generation3 = normDataBack(Generation3,NormMat)
                # Nahrání funkčních hodnot do prvního sloupce
                Generation3.insert(0,'Function values', Results)
            FinalPerfectGeneration = pd.DataFrame(PerfectGeneration)
            #for i in range(0,s):
            #    FinalPerfectGeneration.iloc[:,i+1] = FinalPerfectGeneration.iloc[:,i+1] * NormMat.iloc[i,1] + NormMat.iloc[i,0]
            Generation = Generation3.append(FinalPerfectGeneration.copy(),ignore_index=True)
            if Generation.shape[0] >= r2:
                # Uložení již vyhovujících bodů 
                # selecting rows based on condition 
                GenerationOk = Generation[(Generation['Function values'] >= WantedValuemin) & (Generation['Function values'] <= WantedValuemax)].copy()
                if GenerationOk.shape[0]>0:
                    start = GenerationOk.iloc[0,1:].copy().tolist()
                    T = True
                    break
                print('Set of acceptable points were found. I am trying to find points with desired function values.')

            # Nalezli jsme množinu bodů, které splňují pravděpodobnostní podmínku, nyní musíme splnit podmínku funkčních hodnot    
            # Aplikování genetického algoritmu
            if Generation.shape[0]==0:
                print('Cluster: '+str(ClusterIndex)+'. Start point was not found. I recommend you to increase count of epoch or individuals.')
                break
                #exit()
            Generation,PerfectGeneration2 = Evolution(model,Generation.iloc[:,1:].copy(),WantedValueMean,distance,StoneParameters,MinMax,NormMat,alpha,meanC,CovMatrix,True,'values')
            # Ověří, jestli bod leží v clusteru
            maxVector = ClusterInfo(Generation,PathToClusterDir )
            Generation.insert(0,'Cluster',maxVector[:,2])
            Generation = Generation[Generation['Cluster'] == ClusterIndex ]
            Generation = Generation.drop(Generation.columns[0], axis=1) 
            
            
            E, Generation = CheckPointsInCluster(Generation, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
            if E:
                # Normování dat
                Generation = normData(Generation.copy(),NormMat)
                # Predikce funkčních hodnot k získaným datům pomocí genetického algoritmu
                Results = model.predict(Generation).flatten()   
                # Zpětná transformace znormovaných dat
                Generation = normDataBack(Generation,NormMat)
                # Nahrání funkčních hodnot do prvního sloupce
                Generation.insert(0,'Function values', Results)
                # Ponecháme pouze požadované hodnoty
                Generation = Generation[(Generation['Function values'] >= WantedValuemin) & (Generation['Function values'] <= WantedValuemax)]
            if len(PerfectGeneration2) > 0:
                FinalPerfectGeneration = pd.DataFrame(PerfectGeneration2)
                # Odstranění sloupce funkčních hodnot
                FinalPerfectGeneration = FinalPerfectGeneration.drop(FinalPerfectGeneration.columns[0], axis=1)
                # Ověří, jestli bod leží v clusteru
                maxVector = ClusterInfo(FinalPerfectGeneration,PathToClusterDir )
                FinalPerfectGeneration.insert(0,'Cluster',maxVector[:,2])
                FinalPerfectGeneration = FinalPerfectGeneration[FinalPerfectGeneration['Cluster'] == ClusterIndex ]
                FinalPerfectGeneration = FinalPerfectGeneration.drop(FinalPerfectGeneration.columns[0], axis=1) 
                # Ověření podmínky pravděpodobnosti
                E, FinalPerfectGeneration = CheckPointsInCluster(FinalPerfectGeneration, PathToClusterDir,meanC, CovMatrix,WantedProbmin,WantedProbmax)
                if E:
                    FinalPerfectGeneration = normData(FinalPerfectGeneration.copy(),NormMat)
                    # Predikce funkčních hodnot k získaným datům pomocí genetického algoritmu
                    Results = model.predict(FinalPerfectGeneration).flatten()   
                    # Zpětná transformace znormovaných dat
                    FinalPerfectGeneration = normDataBack(FinalPerfectGeneration,NormMat)
                    # Nahrání funkčních hodnot do prvního sloupce
                    FinalPerfectGeneration.insert(0,'Function values', Results)
                    # Ponecháme pouze požadované hodnoty
                    FinalPerfectGeneration = FinalPerfectGeneration[(FinalPerfectGeneration['Function values'] >= WantedValuemin) & (FinalPerfectGeneration['Function values'] <= WantedValuemax)]
                    Generation = Generation.append(FinalPerfectGeneration, ignore_index=True)
            if GenerationOk.shape[0] > 0:
                Generation = Generation.append(GenerationOk, ignore_index=True)
            if Generation.shape[0] > 0:
                Generation = Generation.sort_values('Function values')
                # Napočítání pravděpodobností     
                prob, pmean = Probability(Generation.iloc[:,1:],meanC,CovMatrix)
                Generation.insert(0,'ProbabilityX/probMean', prob/pmean)
                Generation.insert(0,'ProbabilityX', prob)
                Generation = Generation[(Generation['Function values'] >= WantedValuemin) & (Generation['Function values'] <= WantedValuemax)]
                Generation = Generation[(Generation['ProbabilityX/probMean'] >= WantedProbmin) & (Generation['ProbabilityX/probMean'] <= WantedProbmax)]
                if Generation.shape[0]>0:
                    start = Generation.iloc[0,1:].copy().tolist()
                    T = True
                    break
        if T == False:
            print('Enumeration was not succesfull!')
            continue
        Open.append(start)
        # Zaplnění prvotní generace
        StartGeneration = CreateFirstGeneration(NmbOfChrom,s1,StoneParameters,CovMatrix, meanC, alpha,MinMax)
        StartGeneration.iloc[:,0] = position # Nahrání startovacího bodu 
        # StartGeneration = normData(StartGeneration,NormMat) # Normalizace dat
        # Hledání receptu
        position += MinMax[2,0]
        Name = 0
        Open[0].insert(0, -1) # Rodič
        Open[0].insert(0, Name) # Jméno
        State = Open[0].copy()
        PerfectGeneration = []
        T = True
        while position <= MinMax[1,0]:
            PerfectGeneration = []
            # Vytvoření generací
            Generation = StartGeneration.copy()
            Generation.iloc[:,0] = position #  Nahrání startovacího bodu
            #Generation.iloc[:,0] = (position - NormMat.iloc[0,0]) / NormMat.iloc[0,1]  # Nahrání startovacího bodu
            # Vytvoření daného počtu generací
            for k in range (0,It):
                print(k)
                W = State[2:].copy()
                Generation = Generation_Fitness(model, Generation.copy(), W,NormMat, meanC, CovMatrix, S = 'points')
                # Pokud je hodnota fitness funkce menší rovna distance, pak se jedinec uloží do PerfectGeneration
                probG,probMean = Probability(Generation.iloc[:,2:], meanC, CovMatrix)
                numOfRows = Generation.shape[0]
                #Generation.insert(2,'Prob/Mean', probG/probMean)
                i=0
                while i<numOfRows:
                    if  (abs(Generation.iloc[i,1]- WantedValueMean) <= distance) and (Generation.iloc[i,0] <= PointDistance ) and (abs(probG[i]-WantedDistributionValue) <= DistanceDistributionValue) :
                        #(Generation.iloc[i,1] <= WantedValuemax ) and (Generation.iloc[i,1] >= WantedValuemin) and 
                        PerfectGeneration.append(Generation.iloc[i,1:].copy().tolist())
                        i+=1
                    elif Generation.iloc[i,0] > PointDistance:
                        break
                    else:
                        i+=1
                #Generation = EvolvedGeneration(Generation.iloc[:,1:].copy(),MinMax,NormMat,StoneParameters,CovMatrix, meanC, alpha)
                Generation = EvolvedGeneration(Generation.copy(), MinMax, NormMat, StoneParameters, CovMatrix, meanC, alpha)
                #Generation.iloc[:,0] = (position - NormMat.iloc[0,0]) / NormMat.iloc[0,1]  # Nahrání startovacího bodu
            # Normování dat
            Generation = normData(Generation.copy(),NormMat)
            # Predikce funkčních hodnot pro vytvořenou generaci
            Results = model.predict(Generation).flatten()   
            # Nahrání funkčních hodnot do prvního sloupce
            Generation.insert(0,'Function values', Results)
            # Napočítání vzdálenosti od aktuálního bodu
            # Normování dat
            LastPoint = pd.DataFrame([State[2:]])
            LastPoint = normData(LastPoint,NormMat)
            Fitness = LastPoint - Generation.iloc[:,1:]
            Fitness = np.power(Fitness, 2)
            Fitness = np.sqrt(np.sum(Fitness, axis = 1))
            # Nahrání fitness hodnot do prvního sloupce
            Generation.insert(0,'Fitness',Fitness) 
            # Zpětná transformace znormovaných dat
            Generation.iloc[:,2:] = normDataBack(Generation.iloc[:,2:],NormMat)
            prob, probMean =  Probability(Generation.iloc[:,2:], meanC, CovMatrix) 
            Generation.insert(0,'Prob/Mean', prob/probMean)
            # Vytvoření tabulky pro perfektní množinu
            
            FinalPerfectGeneration = pd.DataFrame(PerfectGeneration)
            if FinalPerfectGeneration.shape[0]>0:
                prob, probMean =  Probability(FinalPerfectGeneration.iloc[:,1:].copy(), meanC, CovMatrix) 
                # Napočítání vzdáleností
                FinalPerfectGeneration.iloc[:,1:] = normData(FinalPerfectGeneration.iloc[:,1:].copy(), NormMat)    
                Fitness = LastPoint - FinalPerfectGeneration.iloc[:,1:]
                Fitness = np.power(Fitness, 2)
                Fitness = np.sqrt(np.sum(Fitness, axis = 1))
                # Zpětná transformace znormovaných dat
                FinalPerfectGeneration.iloc[:,1:] = normDataBack(FinalPerfectGeneration.iloc[:,1:].copy(),NormMat)
                FinalPerfectGeneration.insert(0,'Fitness',Fitness) 
                FinalPerfectGeneration.insert(0,'Prob/Mean', prob/probMean)
                FinalPerfectGeneration.columns = list(Generation.columns.values)
                Generation = Generation.append(FinalPerfectGeneration,ignore_index=True)
            Generation = Generation[(Generation['Function values'] >= WantedValuemin) & (Generation['Function values'] <= WantedValuemax)].copy()
            Generation = Generation[(Generation['Prob/Mean'] >= WantedProbmin*probMean) & (Generation['Prob/Mean'] <= WantedProbmax*probMean)].copy()
            Generation = Generation[(Generation['Fitness'] <= PointDistance)].copy()
            
            # Seřazení potomků od nejhoršího po nejlepšího
            Generation = Generation.sort_values('Fitness',ascending=False)
            r,s = Generation.shape
            T = False
            # Parent = Open[0,1]
            Parent = State[0]
            # Uložení bodu do slovníku Closed
            Closed[State[0]] = State[1:]
            # Nahrání nových expanzivních stavů do Open
            WantedValue = [WantedValuemin, WantedValuemax]
            WantedProb = [WantedProbmin, WantedProbmax]
            for i in range(0,r):
                #if Generation.iloc[i,1] <= WantedValuemax and Generation.iloc[i,1]>=WantedValuemin  and Generation.iloc[i,0] <= PointDistance :
                    # Ověření správnosti nalezené cesty
                    if CheckPath(NormMat,State,Generation.iloc[i,:].copy(),WantedValue,model,meanC,CovMatrix,WantedProb):
                        # Pokud cesta vyhovuje, pak se nalezený bod uloží do seznamu Open na 0. pozici
                        Name += 1
                        NewState = Generation.iloc[i,3:].copy().tolist()
                        NewState.insert(0,Parent) # rodič
                        NewState.insert(0,Name) # jméno
                        Open.insert(1,NewState)
                        T = True
            del Open[0] # Smazání expandujícího stavu ze seznamu Open
            # Pokud jsme nalezli koncový bod, pak jej uložíme do seznamu closed a ukončíme smyčku hledání receptu
            if position == MinMax[1,0] and T:
                State = Open[0].copy()
                Closed[State[0]] = State[1:]
                break
            if T:# Pokud jsme nalezli uzlový bod, pak přejdeme k další pozici, jinak proces zopakujeme
                position += MinMax[2,0]
            elif len(Open)==0:
                print('Recept was not found. I recommend to use another parameters.')
                exit()
            else:    
                #position = State[2] * NormMat[0,1] + NormMat[0,0]
                position = State[2]
            State = Open[0].copy()
            # Smazání prvního (Function values) a nultého (Fitness) sloupce 
            Generation = Generation.drop(Generation.columns[2], axis=1) 
            Generation = Generation.drop(Generation.columns[1], axis=1)
            Generation = Generation.drop(Generation.columns[0], axis=1)
        # Zpětné dohledání nalezené cesty 
        A = list(Closed.keys()) # Vytvoří list klíčů slovníku
        Name = max(A) # Najde maximální klíč
        Path = []
        while Name != -1:
            Path.insert(0,Closed[Name])
            Name = Closed[Name][0]
        # Sestavení podrobného receptu
        Recept = CreateRecept(Path, NormMat, model, meanC, CovMatrix)
        np.savetxt(PathToModel + '/' + 'ReceptCluster'+ str(ClusterIndex) +'.txt', Recept, delimiter=',', newline='\n', header = 'ProbX/ProbMean,Function values,'+H, comments = '')
    
    
    messagebox.showinfo('Glaurung','Konec hledání receptů!')
def GausMixtureModel():
    filename, dataset, H = LoadEntry() 
    X = dataset.iloc[:,1:].copy() 
    #X = dataset.copy()
    #normMat = LoadNormMatrix()
    #X = normData(X,normMat) 
    d = pd.DataFrame(X)
    nmbOfClusters = int( txtGausMixModel.get())
    gmm = GaussianMixture(n_components = nmbOfClusters)
    gmm.fit(d)  
    labels = gmm.predict(d)
    ClusterColumn = 'Cluster'
    d[ClusterColumn]= labels
    Clusters = []
    for i in range(0,nmbOfClusters):
        Clusters.append(d[d[ClusterColumn]== i])
    # Vytvoření složky
    Name=filename+'Clustering'
    os.makedirs('./'+Name)
    s = d.shape[1]-1
    a = np.zeros((nmbOfClusters,s))
    b = np.zeros((s,s))
    dets = np.zeros(nmbOfClusters) 
    for i in range(0,nmbOfClusters):
        b = gmm.covariances_[i].copy()
        print(b)
        rank = np.linalg.matrix_rank(b)
        if rank != b.shape[0]:
            messagebox.showinfo('Glaurung','Covariacne matrix is singular. I recommend to change count of clusters!')
            exit()
        dets[i] = np.linalg.det(b)  
        Points = d.loc[d[ClusterColumn] == i]
        r, s = Points.shape
        P = np.zeros((r,s))
        P = Points.values
        a[i] = gmm.means_[i].copy()
        np.savetxt('./' + Name + '/' + 'PointsInCluster'+str(i)+'.txt', P, delimiter=',', newline='\n', comments='') 
        np.savetxt('./' + Name + '/' + 'CovarianceMatrix'+str(i)+'.txt', b, delimiter=',', newline='\n', comments='')
    np.savetxt('./' + Name + '/' + 'Determinanty'+'.txt', dets, delimiter=',', newline='\n', comments='') 
    np.savetxt('./' + Name + '/' + 'Means.txt', a, delimiter=',', newline='\n', comments='')
    messagebox.showinfo('Glaurung','Covariační matice a střední hodnoty byly vypočítány úspěšně!')
def GaussianClustering():
    filename, data, H = LoadEntry()
    r,s = data.shape
    s = s-1
    M = np.zeros((r,s))
    M = data.iloc[:,1:].copy()
    k= 10
    gamma = np.zeros((r,k))
    ClusterCost = []
    for i in range(1,k):
        # Inicializace počátečních hodnot
        m = np.zeros(i)
        psi = np.full(i, 1/i)
        mu = np.zeros((i,s))
        mu2 = np.zeros((i,s))
        CovarianceMatrixes = []
        for i2 in range(1,i):
            CovMatrix = np.full((s,s),rd.randint(5))
            CovarianceMatrixes.append(CovMatrix)
        initMu = []
        j = 0
        while j < i:
            a = rd.randint(0,r)
            if a in initMu == False or j == 0:
                initMu.append(a)
                m[j] = a
                mu[j,:]=data.iloc[a,1:]
                j += 1
        # EM iteration
        T = True
        while T:
            # Expectation step
            Beta = GaussModel(M, mu,i)
            for j in range(0,r):
                for k2 in range(0,i):
                    gamma[j,k2] = psi[k2] * Beta[j,k2]
                sumation = 0
                for k3 in range(0,i):
                    sumation += psi[k3] * Beta[j,k3]   
                gamma[j,i] = gamma[j,i] / sumation
            # Maximalization step - psi
            psi = np.sum(gamma, axis = 0)
            # Maximalization step - mu, var
            for k2 in range(0,i):
                g = gamma[:,k2]
                sum2 = np.zeros((1,s))
                for j in range(0,r):
                    sum2 += g[j] * M[j,:]
                mu2[k2,:] = sum2/psi[k2]
                # Maximalization step - covariance
                sum2=0
                for j in range(0,r):
                    sum2 += g[j] * np.transpose((M[j,:] - mu[k2,:])) * np.transpose(M[j,:] - mu[k2,:])
                CovarianceMatrixes.append(sum2/psi)
            psi2 = psi/ r
            # Convergence condition
            if np.norm(psi-psi2)<0.01 and np.norm(mu-mu2)<0.01 and np.norm(var-var2)<0.01:
                T = False
            else: 
                psi = psi2
                mu = mu2
                var = var2
        # Cost 
        cost = 0
        for k2 in range(0,r):
            norm = np.norm( M[k2,:] - mu2[:,:]  )
            minNorm = norm.min
            cost += minNorm
        cost /= r
        ClusterCost.append[i,cost]
    print(ClusterCost)
def GaussModel(M,mu,NmbOfClusters):
    r,s = M.shape
    P = np.zeros((r,NmbOfClusters))
 
    for i in range (0,NmbOfClusters):
        for k in range(0,r):
            for j in range(0,s): 
                p = 1/(np.sqrt(2*np.pi)**NmbOfClusters * numpy.linalg.det() ) 
        P[k,i] = p
    return P
def Generation_Fitness(model, Generation, W,NormMat, meanC=None, CovMatrix=None, S = 'values'): # Spočítá Fitness hodnotu pro každého jedince v generaci, seřadí jedince v generaci podle jejich Fitness funkce od nejlepšího po nejhorší a vrtí takto setříděnou generaci
    # Znormování vstupních hodnot
    Generation = normData(Generation,NormMat)
    Results = model.predict(Generation).flatten()
    # Zpětné znormování
    Generation = normDataBack(Generation.copy(),NormMat)
    # Nahrání funkčních hodnot do prvního sloupce
    Generation.insert(0,'Function values', Results)
    # Vybrání nejlepšího jedince
    if S == 'points':
        Generation.iloc[:,1:] = normData(Generation.iloc[:,1:],NormMat)
        # Napočítání vzdálenosti od aktuálního bodu
        W = pd.DataFrame([W])
        W = normData(W,NormMat)
        Fitness2 = W - Generation.iloc[:,1:].copy()
        Fitness2 = np.power(Fitness2, 2)
        sum_row = np.sum(Fitness2, axis = 1)
        Fitness = np.sqrt(sum_row)
        # Zpětné znormování
        Generation.iloc[:,1:] = normDataBack(Generation.iloc[:,1:].copy(),NormMat)
    elif S == 'values':
        Fitness = abs(Results - W)
    elif S == 'probability':
        probability = Probability(Generation.iloc[:,1:], meanC, CovMatrix)
        W = np.full(probability[0].size, W)
        Fitness = abs(probability[0] - W) 
    elif S == 'total':
        PointDistance = W[0] - Generation.iloc[:,1:].copy()
        PointDistance = np.power(Fitness2, 2)
        sum_row = np.sum(Fitness2, axis = 1)
        PointDistance = np.sqrt(sum_row)
        ValuesDistance = abs(Results - W[2])
        probability = Probability(Generation.iloc[:,1:], meanC, CovMatrix)
        W[3] = np.full(probability[0].size, W[3])
        ProbDistance = abs(probability[0] - W[3]) 
        Fitness =  PointDistance + ValuesDistance + ProbDistance
    
    # Nahrání fitness hodnot do prvního sloupce
    Generation.insert(0,'Fitness',Fitness)
    # Seřazení jedinců v generaci od nejlepšího po nejhorší
    Generation = Generation.sort_values('Fitness')
    return Generation
def Generation_Fitness2(model, Generation, Tolerance, NormMat, meanC, CovMatrix, S ):
    r = Generation.shape[0]
    # Znormování vstupních hodnot
    Generation = normData(Generation,NormMat)
    Results = model.predict(Generation).flatten()
    # Zpětné znormování
    Generation = normDataBack(Generation.copy(),NormMat)
    # Nahrání funkčních hodnot do prvního sloupce
    #Generation.insert(0,'Function values', Results)
    Generation2 = pd.DataFrame()
    Generation3 = pd.DataFrame()
    Generation4 = pd.DataFrame()
    FinalGeneration = pd.DataFrame()
    if S == 'ProbValue':
        # Napočítání probability fitness
        # probability = Probability(Generation.iloc[:,1:], meanC, CovMatrix)
        probability = Probability(Generation.iloc[:,:], meanC, CovMatrix)
        W = np.full(probability[0].size, Tolerance[0,0])
        A = abs(probability[0] - W) 
        fitnessProb = A/Tolerance[0,1]
        Generation.insert(0,'fitnessProb', fitnessProb)
        # Napočítání value fitness
        fitnessVal = abs(Results - Tolerance[1,0])
        fitnessVal = fitnessVal/Tolerance[1,1]
        Generation.insert(0,'fitnessVal', fitnessVal)
        # Vytvoření Fitness sloupce
        #print(fitnessProb * fitnessVal)
        # Ponecháme pouze požadované hodnoty
        Generation2 = Generation[(Generation['fitnessProb'] > 1) | (Generation['fitnessVal'] > 1)].copy()
        Generation3 = Generation[(Generation['fitnessProb'] <= 1) & (Generation['fitnessVal'] <= 1)].copy()
        Generation2.insert(0,'Fitness', Generation2['fitnessProb'] + Generation2['fitnessVal'] )
        Generation3.insert(0,'Fitness', Generation3['fitnessProb'] * Generation3['fitnessVal'] )
        # Odstranění sloupce fitnessVal
        Generation2 = Generation2.drop(Generation2.columns[1], axis=1)
        Generation3 = Generation3.drop(Generation3.columns[1], axis=1)
        # Odstranění sloupce fitnessProb
        Generation2 = Generation2.drop(Generation2.columns[1], axis=1)
        Generation3 = Generation3.drop(Generation3.columns[1], axis=1)
        # Spojení předchozích generací
        FinalGeneration = Generation2.copy()
        FinalGeneration = FinalGeneration.append(Generation3, ignore_index=True) 
        '''
        Generation.loc[(Generation.fitnessProb > 1) | (Generation.fitnessVal > 1), 'Fitness'] =  fitnessProb + fitnessVal
        fProb =  np.array(fitnessProb)
        fVal = np.array(fitnessVal)
        Generation.loc[(Generation.fitnessProb <= 1) & (Generation.fitnessVal <= 1), 'Fitness'] =  np.multiply(fProb,fVal)
        '''
    '''
    Fitness = Generation.Fitness.copy().tolist()
    Generation = Generation.drop(Generation.columns[0], axis=1)
    Generation = Generation.drop(Generation.columns[0], axis=1)
    del Generation['Fitness']    
    '''
    # Nahrání fitness hodnot do prvního sloupce
    # Generation.insert(0,'Fitness',Fitness)
    # Normování FinalGenerace
    Generation4 = normData(FinalGeneration.iloc[:,1:].copy(),NormMat)
    Results = model.predict(Generation4).flatten()
    # Seřazení jedinců v generaci od nejlepšího po nejhorší
    FinalGeneration.insert(1,'Function values', Results)
    FinalGeneration = FinalGeneration.sort_values('Fitness')
    # print(Generation)
    # return Generation
    return FinalGeneration
def Iter():
    # Načtení modelu
    PathToFile = filedialog.askopenfilename(initialdir ="C:", title = "Vyber model", filetypes = (("h5 files","*.h5"),("all files","*.*")))
    filename = os.path.basename(PathToFile)
    filename2= PathToFile.replace(filename,'')
    model = keras.models.load_model(PathToFile)
    model.summary()
    # Vstupní iterační data
    DiaOd=int(txtDiaOd.get())
    DiaDo=int(txtDiaDo.get())
    hDia=(DiaDo-DiaOd)/int(txtDiaP.get())
    
    InpOd=int(txtInpOd.get())
    InpDo=int(txtInpDo.get())
    hInp=(InpDo-InpOd)/int(txtInpP.get())
    
    Tz=int(txtTzOd.get())
    TzDo=int(txtTzDo.get())
    hTz=(TzDo-TzOd)/int(txtTzP.get())

    SeedRotOd=int(txtSeedRotOd.get())
    SeedRotDo=int(txtSeedRotDo.get())
    hSeedRot=(SeedRotDo-SeedRotOd)/int(txtSeedRotP.get())

    CruRotOd=int(txtCruRotOd.get())
    CruRotDo=int(txtCruRotDo.get())
    hCruRot=(CruRotDo-CruRotOd)/int(txtCruRotP.get())

    AvgGrowthOd=int(txtAvgGrowthOd.get())
    AvgGrowthDo=int(txtAvgGrowthDo.get())
    hAvgGro=(AvgGrowthDo-AvgGrowthOd)/int(txtAvgGrowthP.get())

    GasFlowOd=int(txtGasFlowOd.get())
    GasFlowDo=int(txtGasFlowDo.get())
    hGasFlow=(GasFlowDo-GasFlowOd)/int(txtGasFlowP.get())
    
    PressureOd=int(txtPressureOd.get())
    PressureDo=int(txtPressureDo.get())
    hPressure=(PressureDo-PressureOd)/int(txtPressureP.get())
    
    LowerCoilOd=int(txtLowerCoilOd.get())
    LowerCoilDo=int(txtLowerCoilDo.get())
    hLowerCoil=(LowerCoilDo-LowerCoilOd)/int(txtLowerCoilP.get())
    
    UpperCoilOd=int(txtUpperCoilOd.get())
    UpperCoilDo=int(txtUpperCoilDo.get())
    hUpperCoil=(UpperCoilDo-UpperCoilOd)/int(txtUpperCoilP.get())

    B=0
    P=0
    As=0
    Sb=0
    if comboPrv.get()=='B': 
        B=1
    elif comboPrv.get()=='P':
        P=1
    elif comboPrv.get()=='As':
        As=1    
    else:
        Sb=1
    Prv=[B,P,As,Sb]
    Or100=0
    Or111=0
    if comboOr.get()=='Or100': 
        Or100=1
    else:
        Or111=1
    Or=[Or100, Or111]
    KA=0
    EKZ=0
    KX=0
    if comboTaz.get()=='KA': 
        KA=1
    elif comboTaz.get()=='EKZ':
        EKZ=1
    else:
        KX=1 
    Dia=DiaOd
    Inp=InpOd
    Tz=TzOd
    SeedRot=SeedRotOd
    CruRot=CruRotOd
    AvgGrowth=AvgGrowthOd
    GasFlow=GasFlowOd
    Pressure=PressureOd
    LowerCoil=LowerCoilOd
    UpperCoil=UpperCoilOd
    if chkRk_state.get():
        Rk=1
    else:
        Rk=0    
    #column_names = ['Oi1',	'Dia',	'B',	'P',	'As',	'Sb',	'Or100',	'Or111','Inp',	'KA',	'EKZ',	'KX',	'TZ',	'RK',	'Seed Rot','	Cru rot',	'Avg growth',	'Gas flow',	'Pressure',	'lower coil','Upper coil']
    
    Dataset=[]
    '''
    A=np.array( [[1, 1,1,1,1,0, 1,1,1,2,20,50,4,5,7,9,50,20,360, 25]])
    Oi1 = model.predict(A)
    print(Oi1)   
    '''
    while Dia<=DiaDo:
        while Inp<=InpDo:
            while Tz<=TzDo:
                while SeedRot<=SeedRotDo:
                    while CruRot<=CruRotDo:
                        while AvgGrowth<=AvgGrowthDo:
                            while GasFlow<=GasFlowDo:
                                while Pressure<=PressureDo:
                                    while LowerCoil<=LowerCoilDo:
                                        while UpperCoil<=UpperCoilDo:
                                            A=np.array([[Dia, B,P,As,Sb,Or100, Or111,Inp,KA,EKZ,KX,Tz,Rk,SeedRot,CruRot,AvgGrowth,GasFlow,Pressure,LowerCoil, UpperCoil]])
                                            Oi1=model.predict(A)
                                            print(Oi1)
                                            print(A)
                                            Dataset.append([Oi1,Dia, B,P,As,Sb,Or100, Or111,Inp,KA,EKZ,KX,Tz,Rk,SeedRot,CruRot,AvgGrowth,GasFlow,Pressure,LowerCoil, UpperCoil])
                                            UpperCoil+=hUpperCoil
                                        UpperCoil=UpperCoilOd
                                        LowerCoil+=hLowerCoil
                                    LowerCoil=LowerCoilOd
                                    Pressure+=hPressure
                                Pressure=PressureOd
                                GasFlow+=hGasFlow
                            GasFlow=GasFlowOd
                            AvgGrowth+=hAvgGro
                        AvgGrowth=AvgGrowthOd
                        CruRot+=hCruRot
                    CruRot=CruRotOd
                    SeedRot+=hSeedRot
                SeedRot=SeedRotOd
                Tz+=hTz
            Tz=TzOd
            Inp+=hInp
        Inp=InpOd
        Dia+=hDia
    # Predikce požadované hodnoty
    np.savetxt(filename2+'IterovaneHodnoty.txt',Dataset ,delimiter=",")
    messagebox.showinfo('Glaurung','Výpočet iterací proběhl úspešně!')
def Info(E,O,L,N,mae,mse,mpe,H,path):
    file1 = open(path,"a+")
    file1.write('---------------Entry parameters of a model------------- '+'\n')
    file1.write('Input features: '+ H+'\n')
    file1.write('Count of hidden layers: '+str(L)+'\n')
    file1.write('Count of neurons: '+str(N)+'\n')
    file1.write('Count of epochs: '+str(txt7.get())+'\n')
    file1.write('Feature normalization: '+str(chkNormalization.get())+'\n')
    file1.write('Normalization by arithmetic mean: '+str(chkMeanMed.get())+'\n')
    file1.write('Batch size: '+str(txtBatchSize.get())+'\n')
    file1.write('Step: '+str(txtStep.get())+'\n')
    file1.write('Train set: '+str(txtTrainSet.get())+'\n')
    file1.write('Loss function: '+comboLossFc.get()+'\n')
    file1.write('---------------Evaluation of a model------------- '+'\n')
    file1.write('Values of errors for test set: '+'\n')
    file1.write('Mean absolute error: '+str(mae)+'\n')
    file1.write('Mean squared error: '+str(mse)+'\n')
    file1.write('Mean absolute percentage error: '+str(mpe)+'%'+'\n')
    file1.write('Mean value of error (error = predicted value - real value): '+str(E)+'\n')
    file1.write('Standart deviation of error: '+str(O)+'\n')
    file1.close()
def Learn(model,train_dataset, train_labels):
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')
    EPOCHS = int(txt7.get()) # Zde volíš, kolikrát se má proces učení opakovat. Pokud je síť nastavená správně, pak by chyba měla konvergovat ke konstantě. Rychlost konvergence a limitu samotnou lze ovlivnit typem neuronové sítě (počet vrstev, neuronů a typem aktivační funkce)
    # Místo pro uložení natrénovaného modelu
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create checkpoint callback
    K='val_loss'
    '''if T:
        K='val_loss'
    else:
        K='val_acc'
    '''
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,monitor=K,save_best_only=True, save_weights_only=True, verbose=0)
    # Napočítání kardinality validační množiny   
    r = train_dataset.shape[0]
    A = r * 100/(float(txtTrainSet.get())*100)
    ValSet = ((1-(float(txtTrainSet.get()))) * A  ) / ( float(txtTrainSet.get()) * 0.01 * A )
    ValSet = ValSet * 0.01
    batchSize = 0
    if int(txtBatchSize.get()) == 0:
        batchSize = r - r * ValSet
    else:
        batchSize = int(txtBatchSize.get())
    history = model.fit(train_dataset, train_labels, batch_size = int(batchSize), epochs = EPOCHS, validation_split = ValSet, verbose=1, callbacks = [cp_callback])
    return history
def LoadAll():
    # Načtení vstupních parametrů
    filename,dataset, H = LoadEntry()
    # Načtení neuronové sítě
    PathToFile = filedialog.askopenfilename(initialdir = "C:", title = "Choose model", filetypes = (("h5 files","*.h5"),("all files","*.*")))
    model = keras.models.load_model(PathToFile)
    Dir = os.path.dirname(PathToFile) # Dir = cesta ke složce z níž načítáme neuronovou síť
    # Načítání normalizační matice
    NormMat = LoadNormMatrix()
    # Zkopírování hodnot z csv souboru do pole MinMax
    r,s = dataset.shape
    r2 = 5
    MinMax = np.zeros((r2,s)) 
    for i in range (0,r):
        MinMax[i]=dataset.loc[i]
    return filename,dataset, H,model,Dir,NormMat,MinMax,s; 
def LoadData(T=True):
    PathToFile = filedialog.askopenfilename(initialdir ="C:", title = "Choose a file of entry data", filetypes = (("csv files","*.csv"),("all files","*.*")))
    filename = os.path.basename(PathToFile)
    data = pd.read_csv(PathToFile, na_values = "?", comment='\t', sep=";", skipinitialspace=True)
    lbl2.configure(text=filename)
    if T:
        return data, filename;
    else:
       return data 
def LoadEntry():
    df2,filename = LoadData()
    dataset = df2.copy()
    H = ''
    for col in dataset.columns:
        H += col+', '
    H=H[:-2]    
    return filename, dataset, H;
def LoadNormMatrix(): # Vrátí normalizační matici
    # Načítání normalizační matice
    PathToNormMat = filedialog.askopenfilename(initialdir = "C:", title = "Vyberte normalizační matici", filetypes = (("txt files","*.txt"),("all files","*.*")))
    NormMat = pd.read_csv(PathToNormMat, sep = ",", header = 0)
    NormMat = pd.DataFrame(NormMat)
    return NormMat
def LoadStoneParameters(s,MinMax): # Načte pevné vstupní parametry
    StoneParameters = {}
    for i in range(0,s):
        if MinMax[3,i] == 1:
            StoneParameters[i] = MinMax[4,i]
    return StoneParameters
def Mutation(j,MinMax,NormMat,CovMatrix, meanC, alpha):
    mean = np.round(meanC, 0)
    diagonal = np.sqrt(np.diagonal(CovMatrix))# směrodatná odchylka
    variances = np.round(alpha * diagonal, 0) 
    variance2 = alpha * diagonal
    if MinMax[2,j] == 1:
        RndValue = rd.randint( mean[j] -  variances[j], mean[j] + variances[j]   )  
        #RndValue = rd.randint(MinMax[0,j],MinMax[1,j])
        #Generation2 = (RndValue-NormMat.iloc[j,0])/NormMat.iloc[j,1] # Normalizace zmutovaného parametru
    elif MinMax[2,j] == 0:
        line = rd.randint(0,1)
        RndValue = MinMax[line,j]
        #Generation2 = (MinMax[line,j]-NormMat.iloc[j,0])/NormMat.iloc[j,1] # Normalizace zmutovaného parametru
    else:
        Sequence = list(np.arange( meanC[j] - variance2[j], meanC[j] + variance2[j], variance2[j]*0.01))# MinMax[0,j],MinMax[1,j],MinMax[2,j]))
        # Sequence = list(np.arange(MinMax[0,j],MinMax[1,j],MinMax[2,j]))
        #RndValue = (rd.choice(Sequence)-NormMat.iloc[j,0])/NormMat.iloc[j,1] # Normalizace zmutovaného parametru
        #Generation2 = RndValue 
        RndValue = rd.choice(Sequence)
    return RndValue
def MultGausDistribution(S,mu,X): # S = kovariační matice, mu = vektor středních hodnot, X matice vstupních dat ke kterým hledáme hodnotu pravděpodobnosti
    r,s = S.shape
    det = np.linalg.det(S) # Determinant kovariační matice
    k = 1/((2*np.pi)^(s/2)*np.sqrt(det)) # Koeficient pro pravděpodobnostní funkci
    u = X[:] - mu
    pX = k*np.exp(-0.5 * u * S * np.transpose(u)) # Pravděpodobnostní hodnota pro každý vstupní parametr
    return pX
def norm(data,T):
    r,s = data.shape
    if T:
        A = np.zeros((s, 2))
    for i in range(0,s):
        if data.iloc[:,i].min() != 0 or data.iloc[:,i].max() != 1:
            if chkMeanMed.get():
                m = data.iloc[:,i].mean()
                std = data.iloc[:,i].std()
            else:    
                m = data.iloc[:,i].median()
                A2 = np.array(data.iloc[:,i]).reshape((1,r))
                B = np.zeros((1,r))
                B = m
                C = np.power(A2-B, 2)
                std = np.sqrt( (1/r) * np.sum(C) )
            if T:
                A[i,0] = m 
                A[i,1] = std
            data.iloc[:,i] = (data.iloc[:,i]-m)/std
        else:
            if T:
                A[i,1] = 1
    if T:
        return data, A;
    else:
        return data
def normData(data,NormMat):
    s = len(data.columns) # počet sloupců
    for i in range (0,s):
        if data.iloc[:,i].min() != 0 or data.iloc[:,i].max() != 1:
            data.iloc[:,i] = (data.iloc[:,i]-NormMat.iloc[i,0])/NormMat.iloc[i,1]
    return data
def normDataBack(data, NormMat):
    s = len(data.columns) # počet sloupců
    for i in range(0,s):
        data.iloc[:,i] = data.iloc[:,i] * NormMat.iloc[i,1] + NormMat.iloc[i,0]
    return data
def plot_history(history,Name):
    # Mean Abs Error
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.legend()
    plt.savefig('./'+Name+'/'+'Mean Abs Error.png', bbox_inches='tight')
    # Mean Square Error
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.savefig('./'+Name+'/'+'Mean Square Error.png', bbox_inches='tight')
    # Mean Absolute Percentage Error
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Percentage Error [%]')
    plt.plot(hist['epoch'], hist['mean_absolute_percentage_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_percentage_error'], label = 'Val Error')
    plt.legend()
    plt.savefig('./'+Name+'/'+'Mean Absolute Percentage Error.png', bbox_inches='tight')
    '''
    if T:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
        plt.legend()
        plt.savefig('./'+Name+'/'+'Mean Abs Error.png', bbox_inches='tight')
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
        plt.legend()
        plt.savefig('./'+Name+'/'+'Mean Square Error.png', bbox_inches='tight')
    else:
        history_dict = history.history
        history_dict.keys()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        # "bo" is for "blue dot"
        # Graf pro ztrátovou funkcí tj. loss
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./'+Name+'/'+'Loss.png', bbox_inches='tight')
        # Graf pro accuracy
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('./'+Name+'/'+'Accuracy.png', bbox_inches='tight')
    '''
def Predict():
    # Načtení dat
    data = pd.DataFrame()
    TState = chkCsv_state.get()
    if TState:
        data = LoadData(False)
    else:
        '''
        B=0
        P=0
        if comboPrv.get()=='B': 
            B=1
        elif comboPrv.get()=='P':
            P=1
        Or100=0
        Or111=0
        if comboOr.get()=='Or100': 
            Or100=1
        else:
            Or111=1
        KA=0
        KX=0
        if comboTaz.get()=='KA': 
            KA=1
        else:
            KX=1 
        Rk=0
        if chkRk_state.get():
            Rk=1
        '''    
        #d = [[float(txtPoz.get()),float(txtDia.get()),B,P,Or100,Or111,float(txtInp.get()),KA,KX,float(txtTz.get()),float(Rk),float(txtSeedRot.get()),float(txtCruRot.get()),float(txtAvgGrowth.get()),float(txtGasFlow.get()),float(txtPressure.get()),float(txtLowerCoil.get()), float(txtUpperCoil.get())]]
        d = [[float(txtPoz.get()),float(txtInp.get()),float(txtSeedRot.get()),float(txtCruRot.get()),float(txtGasFlow.get()),float(txtPressure.get()),float(txtLowerCoil.get()), float(txtUpperCoil.get())]]
        data = pd.DataFrame(d)
    T = chkRegrese_state.get()
    # Pokud znám funkční hodnotu pak se první sloupec zkopíruje do test_labels a následně smaže z data
    H = ''
    if T:
        H = data.columns[0]+', '
        test_labels = data.iloc[:,0]
        data = data.drop(data.columns[0],axis=1)    
    if TState:
        for col in data.columns:
            H += col + ', '
        H = H[:-2] 
    else:
        H = 'Pozition, Input, SeedRot, CruRot,GasFlow, Presssure, LowerCoil, UpperCoil'
    # Ověření podmínky interpolace
    PathToClusterDir = filedialog.askdirectory(title = 'Choose a file of clusters')
    DistValueMin = float(txtProbmin.get())
    DistValueMax = float(txtProbmax.get())
    ValidData, data = CheckPoints(data,PathToClusterDir,DistValueMin,DistValueMax)
    if ValidData == False:
        messagebox.showinfo('Glaurung','Intput points do not satisfy distribution contrains!. I recommend to change the constrain.')
        exit()
    # Načtení složky
    PathToModel = filedialog.askdirectory(title = 'Choose a file of model')
    # Načítání normalizační matice a normování dat
    if chkNormalization.get():
        NormMat = pd.read_csv(PathToModel+'/NormalizacniMatice.txt', sep = ",", header = 0)   
        NormMat = pd.DataFrame(NormMat) 
        data = normData(data,NormMat)
    # Načtení modelu/ů
    DirName = os.path.basename(PathToModel)
    # Načtení neuronové sítě
    model = keras.models.load_model(PathToModel +'/'+DirName+'.h5')
    # Dimenze matice vstupních dat
    r = data.shape[0]
    A = np.zeros((r, 1))
    # Predikce požadované hodnoty
    Results = model.predict(data).flatten()
    A[:,0] = Results # Uložení predikovaných hodnot do i. sloupce
    # Graf zkutečných a predikovaných hodnot
    data = normDataBack(data,NormMat) #zpětné znormování dat
    # Zjistí z jakého clusteru daný bod je a vypíše jeho pravděpodobnost a p/pmean
    maxVector = ClusterInfo(data,PathToClusterDir)
    if T:
        data.insert(0, "Original value", test_labels)
    data.insert(0, "Predicted value", A[:,0]) # vložení funkčních hodnot do prvního sloupce   
    data.insert(0,'Cluster',maxVector[:,2])
    data.insert(0,'pX/pMean ',maxVector[:,1])
    data.insert(0,'Dist value ',maxVector[:,0])
    # Uložení dat
    np.savetxt(PathToModel + '/' + 'PredictedValues.txt', data, delimiter=',', newline='\n', header = 'Dist value, Dist value/pmean,Cluster,Predicted value,' + H, comments = '')
    messagebox.showinfo('Glaurung','Algorithm was finished succesfully!')
def Probability(x,meanC,CovMatrix): # Spočte pravděpodobnost každého bodu v množině x s níž bod náleží k příslušnému clusteru
    if x.size == 0:
        D = np.sqrt( np.linalg.det(CovMatrix))
        s = meanC.size
        t4 = pow(2 * np.pi, s * 0.5) * D
        pmean = 1/t4
        return pmean
    x = x.values.copy()
    C = np.linalg.inv(CovMatrix)
    D = np.sqrt( np.linalg.det(CovMatrix))
    r, s = x.shape
    x = np.transpose(x)
    meanC = meanC.values.reshape((1,s))
    meanC = np.tile(meanC,(r,1))
    meanC = np.transpose(meanC)    
    t = x-meanC
    px = np.zeros(r)
    t4 = pow(2 * np.pi, s * 0.5) * D
    # Napočítání pravděpodobnsoti, že daný bod patří k příslušnému clusteru
    # Cluster je určen hodnotami mean a CovMatrix
    for i in range(0,r):
        t2 = np.matmul(np.transpose(t[:,i]),C).reshape((1,s))
        t3 = np.matmul(t2,t[:,i])
        px[i] = 1/t4 * np.exp(-0.5 * t3) 
    pmean = 1/t4
    return px,pmean;
def Regrese(X,Y,NameOfFolder, NameOfFile=""):
    plt.figure()
    plt.scatter(X, Y)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    minValue = 0
    if Y.min() > X.min():
        minValue = X.min()  #Přičtení 2 zaručí, že bod nebude ležet na kraji grafu tzn. přidáme padding
    else:
        minValue = Y.min()  #Přičtení 2 zaručí, že bod nebude ležet na kraji grafu tzn. přidáme padding
    maxValue = 0
    if Y.max() < X.max():
        maxValue = X.max()  #Přičtení 2 zaručí, že bod nebude ležet na kraji grafu tzn. přidáme padding
    else:
        maxValue = Y.max()  #Přičtení 2 zaručí, že bod nebude ležet na kraji grafu tzn. přidáme padding
    #plt.xlim([minValue,plt.xlim()[1]])
    #plt.ylim([minValue,plt.ylim()[1]])
    plt.xlim([minValue,maxValue])
    plt.ylim([minValue,maxValue])
    #plt.autoscale(tight=False)
    #plt.margins(x=0.2,y=0.2)
    plt.plot([minValue,maxValue], [minValue, maxValue])
    plt.savefig(NameOfFolder+'/'+'Prediction'+ NameOfFile +'.png', bbox_inches='tight')
def Rezidua(Prediction, RealValue):
    res = Prediction - RealValue
    return res.mean(), res.std();
def Interpolace():
    PathToFile = filedialog.askopenfilename(initialdir ="C:", title = "Select file", filetypes = (("csv files","*.csv"),("all files","*.*")))
    filename = os.path.basename(PathToFile)
    df2 = pd.read_csv(PathToFile, na_values = "?", comment='\t', sep=";", skipinitialspace=True)
    r,s = df2.shape
    dataset = df2.copy()
    dataset2 = dataset.copy()
    train_dataset = dataset.drop(dataset.columns[0],axis=1)
    #train_dataset = train_dataset2.drop(train_dataset2.columns[0],axis=1)
    # Oddělení názvů krystalů
    # Normalizace trénovacích dat
    NormTrain_dataset = norm(train_dataset, False)
    sNorm = s - 1 # Počet sloupců je v NormTrain_dataset menší o 1 (byl odstraněn sloupec funkčních hodnot)
    #sNorm = s - 2 # Počet sloupců je v NormTrain_dataset menší o 2 (byl odstraněn sloupec funkčních hodnot a sloupec krystalů)
    N = r
    pocetUzlu = int(txtUzly.get())
    eps = float(txtEps.get())
    w = pocetUzlu-1
    delkaKroku = 1/w
    #B = np.zeros((s,2))
    B = np.zeros((s-1,2))
    V = df2.columns.values
    H = V[0]
    for i in range(1, s):
        H += ',' + V[i]
    for i in range(0,r-1):
        print(i)
        for j in range(i+1,r):
           # T = False
            
            T = True
            for k in range(0,sNorm):
            # Ověření, zda jsi sou dva body dostatečně blízké
                if abs(NormTrain_dataset.iloc[i,k]-NormTrain_dataset.iloc[j,k])>=eps:
                    T = False
                    break
                
           # if ( abs(train_dataset.iloc[i,0] - train_dataset.iloc[j,0] ) <= 350  ) and Krystal[i]==Krystal[j]:
           #     T = True
            if T:
                # Tvorba matice počátečních bodů a vektoru posunutí
                #B[:,0] = train_dataset2.iloc[i,:]
                #B[:,1] = train_dataset2.iloc[j,:] - B[:,0] # Vektor posuvu pro u. souřadnici
                B[:,0] = dataset.iloc[i,:]
                B[:,1] = dataset.iloc[j,:] - B[:,0] # Vektor posuvu pro u. souřadnici
                # Interpolace dat po spojnici bodů
                for k in range(1, w):    
                    dataset2.loc[N] = B[:,0] + delkaKroku * k * B[:,1]
                    N += 1
    np.savetxt(filename+'InterpoloceEpsilon'+str(eps)+'Uzlu' +str(pocetUzlu) + '.txt', dataset2, delimiter=',', newline='\n', header = H , comments='')
    messagebox.showinfo('Glaurung','Interpolace dat proběhla úspěšně!')
    
tf.enable_eager_execution()
window = Tk()
window.title("Glaurung")
window.geometry('710x540')
#window.iconbitmap(r'C:\Users\zbjbgd\Desktop\Glaurung2\dragon.ico')
# Parametry neuronové sítě
rP=0
lblOd = Label(window, text="Min")
lblOd.grid(column=1, row=rP)
lblDo = Label(window, text="Max")
lblDo.grid(column=2, row=rP)
rP+=1
lbl5 = Label(window, text="Count of hidden layers:")
lbl5.grid(column=0, row=rP)
txtHLOd = Entry(window,width=10)
txtHLOd.grid(column=1, row=rP)
txtHLDo = Entry(window,width=10)
txtHLDo.grid(column=2, row=rP)
rP+=1
lbl6 = Label(window, text="Count of neurons:")
lbl6.grid(column=0, row=rP)
txtNOd = Entry(window,width=10)
txtNOd.grid(column=1, row=rP)
txtNDo = Entry(window,width=10)
txtNDo.grid(column=2, row=rP)
rP+=1
lbl7 = Label(window, text="Count of epochs:")
lbl7.grid(column=0, row=rP)
txt7 = Entry(window,width=10)
txt7.grid(column=1, row=rP)
rP+=1
lblRestart = Label(window, text="Count of restart :")
lblRestart.grid(column=0, row=rP)
txtRestart = Entry(window,width=10)
txtRestart.grid(column=1, row=rP)

rP+=1
btn = Button(window, text="Train neural network", command = Build )
btn.grid(column=0, row=rP)
'''
chkRegrese_Klasifikace = BooleanVar()
chkRegrese_Klasifikace.set(True) #set check state
chkRegrese = Checkbutton(window, text='Regrese/Klasifikace', var=chkRegrese_Klasifikace)
chkRegrese.grid(column=1, row=rP)
'''
chkNormalization = BooleanVar()
chkNormalization.set(True) #set check state
chkNorm = Checkbutton(window, text='Normalize data', var=chkNormalization)
chkNorm.grid(column=1, row=rP)

chkMeanMed = BooleanVar()
chkMeanMed.set(True) #set check state
chkMeanMedian = Checkbutton(window, text='Mean average/median', var=chkMeanMed)
chkMeanMedian.grid(column=2, row=rP)
'''
chkPCA = BooleanVar()
chkPCA.set(False) #set check state
chkPCA2 = Checkbutton(window, text='PCA', var=chkPCA)
chkPCA2.grid(column=1, row=rP)
'''
rP+=1
# Nastavení hyperparametrů
lblBatchSize = Label(window, text="Batch size:")
lblBatchSize.grid(column=1, row = rP)
txtBatchSize = Entry(window,width=10)
txtBatchSize.grid(column=2, row=rP)
txtBatchSize.insert(0,'32')
rP+=1
lblStep = Label(window, text="Step:")
lblStep.grid(column=1, row = rP)
txtStep = Entry(window,width=10)
txtStep.grid(column=2, row=rP)
txtStep.insert(0,'0.001')
rP+=1
lblTrainSet = Label(window, text="Train set:")
lblTrainSet.grid(column=1, row = rP)
txtTrainSet = Entry(window,width=10)
txtTrainSet.grid(column=2, row=rP)
txtTrainSet.insert(0,'0.9')

rP+=1
lblLossFc = Label(window, text="Loss function")
lblLossFc.grid(column=1, row=rP)
comboLossFc = Combobox(window, width=7)
comboLossFc['values'] = ('MAE','MSE','MAPE')
comboLossFc.current(0) #set the selected item
comboLossFc.grid(column=2, row = rP)

rP+=1
btn2 = Button(window, text="Calculate result", command = Predict)
btn2.grid(column=0, row=rP)
chkRegrese_state = BooleanVar()
chkRegrese_state.set(True) #set check state
chkRegrese = Checkbutton(window, text='Output value is known/unkown', var=chkRegrese_state)
chkRegrese.grid(column=1, row=rP)
chkCsv_state = BooleanVar()
chkCsv_state.set(True) #set check state
chkExcel = Checkbutton(window, text='Read data from csv/cells', var=chkCsv_state)
chkExcel.grid(column=2, row=rP)



# Najdi parametry - genetický algoritmus
rP+=1
btnFind = Button(window, text="Find features", command = FindParameters)
btnFind.grid(column=0, row = rP)
lblFindJed = Label(window, text="Count of individuals:")
lblFindJed.grid(column=1, row = rP)
txtFindJed = Entry(window,width=10)
txtFindJed.grid(column=2, row=rP)
rP+=1
lblFindIter = Label(window, text="Count of epochs:")
lblFindIter.grid(column=1, row = rP)
txtFindIter = Entry(window,width=10)
txtFindIter.grid(column=2, row=rP)
rP+=1
lblFindFcemin = Label(window, text="Min value of output:")
lblFindFcemin.grid(column=1, row = rP)
txtFindFcemin = Entry(window,width=10)
txtFindFcemin.grid(column=2, row=rP)
rP+=1
lblFindFcemax = Label(window, text="Max value of output:")
lblFindFcemax.grid(column=1, row = rP)
txtFindFcemax = Entry(window,width=10)
txtFindFcemax.grid(column=2, row=rP)
rP+=1
lblProbmin = Label(window, text="Min distribution value:")
lblProbmin.grid(column=1, row = rP)
txtProbmin = Entry(window,width=10)
txtProbmin.grid(column=2, row=rP)
rP+=1
lblProbmax = Label(window, text="Max distribution value:")
lblProbmax.grid(column=1, row = rP)
txtProbmax = Entry(window,width=10)
txtProbmax.grid(column=2, row=rP)
rP+=1
lblClusterMin = Label(window, text="Index of first cluster:")
lblClusterMin.grid(column=1, row = rP)
txtClusterMin = Entry(window,width=10)
txtClusterMin.grid(column=2, row=rP)
rP+=1
lblClusterMax = Label(window, text="Index of last cluster:")
lblClusterMax.grid(column=1, row = rP)
txtClusterMax = Entry(window,width=10)
txtClusterMax.grid(column=2, row=rP)


# Najdi cestu - recept
rP+=1
btnFindRecept = Button(window, text="Find recept", command = FindRecept)
btnFindRecept.grid(column=0, row = rP)
lblFindRecept = Label(window, text="Max distance of points:")
lblFindRecept.grid(column=1, row = rP)
txtFindRecept = Entry(window,width=10)
txtFindRecept.grid(column=2, row=rP)

# Parametry iteračního procesu
a=0
s1=3
s2=s1+1
a=a+1
lblPoz = Label(window, text="Pozition")
lblPoz.grid(column=s1, row=a)
txtPoz = Entry(window,width=10)
txtPoz.grid(column=s2, row=a)
'''a=a+1
lblDia = Label(window, text="Dia")
lblDia.grid(column=s1, row=a)
txtDia = Entry(window,width=10)
txtDia.grid(column=s2, row=a)

a=a+1
lblPrv = Label(window, text="Prvek")
lblPrv.grid(column=s1, row=a)
comboPrv = Combobox(window, width=7)
comboPrv['values']= ('B','P')
comboPrv.current(0) #set the selected item
comboPrv.grid(column=s2, row=a)

a=a+1
lblOr = Label(window, text="Or")
lblOr.grid(column=s1, row=a)
comboOr = Combobox(window, width=7)
comboOr['values']= ('Or100','Or111')
comboOr.current(0) #set the selected item
comboOr.grid(column=s2, row=a)
'''
a=a+1
lblInp = Label(window, text="Inp")
lblInp.grid(column=s1, row=a)
txtInp = Entry(window,width=10)
txtInp.grid(column=s2, row=a)
'''a=a+1
lblTazicka = Label(window, text="Tažička")
lblTazicka.grid(column=s1, row=a)
comboTaz = Combobox(window, width=7)
comboTaz['values']= ('KA','KX')
comboTaz.current(0) #set the selected item
comboTaz.grid(column=s2, row=a)
a+=1
lblTz = Label(window, text="Tz")
lblTz.grid(column=s1, row=a)
txtTz = Entry(window,width=10)
txtTz.grid(column=s2, row=a)
a=a+1
lblRk = Label(window, text="Rk")
lblRk.grid(column=s1, row=a)
chkRk_state = BooleanVar()
chkRk_state.set(True) #set check state
chkRk = Checkbutton(window, text='Ano/Ne', var=chkRk_state)
chkRk.grid(column=s2, row=a)
'''
a+=1
lblSeedRot = Label(window, text="Seed Rot")
lblSeedRot.grid(column=s1, row=a)
txtSeedRot = Entry(window,width=10)
txtSeedRot.grid(column=s2, row=a)
a=a+1
lblCruRot = Label(window, text="Cru Rot")
lblCruRot.grid(column=s1, row=a)
txtCruRot = Entry(window,width=10)
txtCruRot.grid(column=s2, row=a)
'''
a=a+1
lblAvgGrowth = Label(window, text="Avg growth")
lblAvgGrowth.grid(column=s1, row=a)
txtAvgGrowth = Entry(window,width=10)
txtAvgGrowth.grid(column=s2, row=a)
'''
a=a+1
lblGasFlow = Label(window, text="Gas flow")
lblGasFlow.grid(column=s1, row=a)
txtGasFlow = Entry(window,width=10)
txtGasFlow.grid(column=s2, row=a)
a=a+1
lblPressure = Label(window, text="Pressure")
lblPressure.grid(column=s1, row=a)
txtPressure = Entry(window,width=10)
txtPressure.grid(column=s2, row=a)
a=a+1
lblLowerCoil = Label(window, text="Lower coil")
lblLowerCoil.grid(column=s1, row=a)
txtLowerCoil = Entry(window,width=10)
txtLowerCoil.grid(column=s2, row=a)
a=a+1
lblUpperCoil = Label(window, text="Upper coil")
lblUpperCoil.grid(column=s1, row=a)
txtUpperCoil = Entry(window,width=10)
txtUpperCoil.grid(column=s2, row=a)
a=a+1




# Interpolace dat
rP+=1
'''btnInt = Button(window, text="Interpolace dat", command = Interpolace)
btnInt.grid(column=0, row=rP)
'''
lblUzly = Label(window, text="Count points of interpolation:")
lblUzly.grid(column=1, row=rP)
txtUzly = Entry(window,width=10)
txtUzly.grid(column=2, row=rP)
'''rP+=1
lblEps = Label(window, text="Testovací kritérium:")
lblEps.grid(column=1, row=rP)
txtEps = Entry(window,width=10)
txtEps.grid(column=2, row=rP)
'''
# Gauss mixture model
rP+=1
btnClustering = Button(window, text="Clustering", command = Clustering)
btnClustering.grid(column=0, row=rP)
lblMaxClusters = Label(window, text="Max count of clusters:")
lblMaxClusters.grid(column=1, row=rP)
txtMaxClusters = Entry(window,width=10)
txtMaxClusters.grid(column=2, row=rP)


# Gauss mixture model
rP+=1
btnGausMixModel = Button(window, text="Gauss mixture model", command = GausMixtureModel)
btnGausMixModel.grid(column=0, row=rP)
lblGausMixModel = Label(window, text="Count of clusters:")
lblGausMixModel.grid(column=1, row=rP)
txtGausMixModel = Entry(window,width=10)
txtGausMixModel.grid(column=2, row=rP)

# Info
rP +=2
lbl = Label(window, text="Choosen file: ")
lbl.grid(column=0, row=rP)
lbl2 = Label(window, text="")
lbl2.grid(column=1, row=rP)
window.mainloop()