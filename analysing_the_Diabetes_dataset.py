# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:26:40 2023

@author: szyns
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error

data = load_diabetes()
df = pd.DataFrame(data.data,columns=data.feature_names)
df_corr=df.corr()
fig,ax = plt.subplots(1,5,figsize=(30,10))

#korelacje

ax[0].set_title('Wykres zaleznosci wieku od bmi: ')
ax[0].scatter(df['bmi'],df['age'])
ax[0].set_xlabel('bmi')
ax[0].set_ylabel('wiek')
ax[1].set_title('Wykres zaleznosci bp od bmi: ')
ax[1].scatter(df['bmi'],df['bp'])
ax[1].set_xlabel('bmi')
ax[1].set_ylabel('bp')
ax[2].set_title('Wykres zaleznosci s5 od bmi: ')
ax[2].scatter(df['bmi'],df['s5'])
ax[2].set_xlabel('bmi')
ax[2].set_ylabel('s5')
ax[3].set_title('Wykres zaleznosci s1 od s2: ')
ax[3].scatter(df['s2'],df['s1'])
ax[3].set_xlabel('s2')
ax[3].set_ylabel('s1')
ax[4].set_title('Wykres zaleznosci s2 od s4: ')
ax[4].scatter(df['s4'],df['s2'])
ax[4].set_xlabel('s4')
ax[4].set_ylabel('s2')

#wielokrotne testowanie modelu

data_cechy = df.columns.to_list()
data_arr= df.values
X,y= data_arr[:,:-1], data_arr[:,-1]

def funkcja(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) 
    for i in range(int(liczba_powtorzen/(liczba_powtorzen/10))):
        for j in range(math.ceil(liczba_powtorzen/10)):
            if(math.ceil(liczba_powtorzen/10)==1):
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                linReg=LinearRegression()
                linReg.fit(X_train,y_train)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                   ax[0].scatter(y_test,y_pred)
                   ax[0].plot([minval,maxval],[minval,maxval])
                   ax[0].set_xlabel('y_test')
                   ax[0].set_ylabel('y_pred')
                   ax[0].set_title('Test nr 1')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                 
                   
            else:
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                linReg=LinearRegression()
                linReg.fit(X_train,y_train)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                    ax[0].scatter(y_test,y_pred)
                    ax[0].plot([minval,maxval],[minval,maxval])
                    ax[0].set_xlabel('y_test')
                    ax[0].set_ylabel('y_pred')
                    ax[0].set_title('Test nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1


print('Sredni procent bledu regresji: ',funkcja(1000))

#uwzglednienie wartosci odstajacych przez ich usunięcie
def funkcja1(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) 
    for i in range(int(liczba_powtorzen/(liczba_powtorzen/10))):
        for j in range(math.ceil(liczba_powtorzen/10)):
            if(math.ceil(liczba_powtorzen/10)==1):
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                X_train_no_outliers = X_train[~outliers,:]
                y_train_no_outliers = y_train[~outliers]
                linReg=LinearRegression()
                linReg.fit(X_train_no_outliers,y_train_no_outliers)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                   ax[0].scatter(y_test,y_pred)
                   ax[0].plot([minval,maxval],[minval,maxval])
                   ax[0].set_xlabel('y_test')
                   ax[0].set_ylabel('y_pred')
                   ax[0].set_title('Test nr 1')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                 
                   
            else:
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                X_train_no_outliers = X_train[~outliers,:]
                y_train_no_outliers = y_train[~outliers]
                linReg=LinearRegression()
                linReg.fit(X_train_no_outliers,y_train_no_outliers)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                    ax[0].scatter(y_test,y_pred)
                    ax[0].plot([minval,maxval],[minval,maxval])
                    ax[0].set_xlabel('y_test')
                    ax[0].set_ylabel('y_pred')
                    ax[0].set_title('Test nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1


print('Sredni procent bledu regresji po usunieciu wartosci odstajacych: ',funkcja1(1000))

#uwzglednienie wartosci odstajacych przez ich zastapienie
def funkcja2(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) 
    for i in range(int(liczba_powtorzen/(liczba_powtorzen/10))):
        for j in range(math.ceil(liczba_powtorzen/10)):
            if(math.ceil(liczba_powtorzen/10)==1):
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                y_train_mean = y_train.copy()
                y_train_mean[outliers]=y_train.mean()
                linReg=LinearRegression()
                linReg.fit(X_train,y_train_mean)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                   ax[0].scatter(y_test,y_pred)
                   ax[0].plot([minval,maxval],[minval,maxval])
                   ax[0].set_xlabel('y_test')
                   ax[0].set_ylabel('y_pred')
                   ax[0].set_title('Test nr 1')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                 
                   
            else:
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                y_train_mean = y_train.copy()
                y_train_mean[outliers]=y_train.mean()
                linReg=LinearRegression()
                linReg.fit(X_train,y_train_mean)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                    ax[0].scatter(y_test,y_pred)
                    ax[0].plot([minval,maxval],[minval,maxval])
                    ax[0].set_xlabel('y_test')
                    ax[0].set_ylabel('y_pred')
                    ax[0].set_title('Test nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1


print('Sredni procent bledu regresji po zastapieniu wartosci odstajacyh srednia odstajacych: ',funkcja2(1000))

#Generacja nowych cech
nowe_dane = np.stack( [X[:, 7]*X[:,2],X[:, 7]*X[:, 2],X[:, 6]/X[:, 1],X[:, 7]/X[:, 1],X[:, 3]*X[:, 6],X[:, 0]/X[:, 4],X[:, 8]/X[:, 7]], axis=-1)
X_additional = np.concatenate([X, nowe_dane], axis=-1)

def funkcja3(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) 
    for i in range(int(liczba_powtorzen/(liczba_powtorzen/10))):
        for j in range(math.ceil(liczba_powtorzen/10)):
            if(math.ceil(liczba_powtorzen/10)==1):
                X_train, X_test, y_train, y_test  = train_test_split(X_additional,y,test_size = 0.2, shuffle=True)
                linReg=LinearRegression()
                linReg.fit(X_train,y_train)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                   ax[0].scatter(y_test,y_pred)
                   ax[0].plot([minval,maxval],[minval,maxval])
                   ax[0].set_xlabel('y_test')
                   ax[0].set_ylabel('y_pred')
                   ax[0].set_title('Test nr 1')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                 
                   
            else:
                X_train, X_test, y_train, y_test  = train_test_split(X_additional,y,test_size = 0.2, shuffle=True)
                linReg=LinearRegression()
                linReg.fit(X_train,y_train)
                y_pred = linReg.predict(X_test)
                minval = min(y_test.min(),y_pred.min())
                maxval = max(y_test.max(),y_pred.max())
                if((i)==0):
                    ax[0].scatter(y_test,y_pred)
                    ax[0].plot([minval,maxval],[minval,maxval])
                    ax[0].set_xlabel('y_test')
                    ax[0].set_ylabel('y_pred')
                    ax[0].set_title('Test nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test nr '+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1


print('Sredni procent bledu regresji po dodaniu nowych danych: ',funkcja(1000))
