# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:19:21 2023

@author: szyns
"""
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

data = pd.read_excel('practice_lab_2.xlsx')
data_cechy = data.columns.to_list()
data_arr= data.values
X,y= data_arr[:,:-1], data_arr[:,-1]

from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2,random_state=221, shuffle=True)

linReg=LinearRegression()
linReg.fit(X_train,y_train)
y_pred = linReg.predict(X_test)
minval = min(y_test.min(),y_pred.min())
maxval = max(y_test.max(),y_pred.max())
#plt.scatter(y_test,y_pred)
#plt.plot([minval,maxval],[minval,maxval])
#plt.xlabel('y_test')
#plt.ylabel('y_pred')

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error (y_test,y_pred)

def funkcja1(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) #plt.subplots(int(liczba_powtorzen/(liczba_powtorzen/10)),math.ceil(liczba_powtorzen/10),figsize=(15,10))
    for i in range(int(liczba_powtorzen/(liczba_powtorzen/10))):
        for j in range(math.ceil(liczba_powtorzen/10)):
            if(math.ceil(liczba_powtorzen/10)==1):
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                X_train_no_outliers = X_train[~outliers,:]
                y_train_no_outliers = y_train[~outliers]
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
                   ax[0].set_title('Test z usunieciem danych dostajacych nr 1')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test z usunieciem danych dostajacych nr '+str(liczba_powtorzen))
                    break
                 
                   
            else:
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                X_train_no_outliers = X_train[~outliers,:]
                y_train_no_outliers = y_train[~outliers]
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
                    ax[0].set_title('Test z usunieciem danych dostajacych nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test z usunieciem danych dostajacych nr '+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1

def funkcja2(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) #plt.subplots(int(liczba_powtorzen/(liczba_powtorzen/10)),math.ceil(liczba_powtorzen/10),figsize=(15,10))
    for i in range(int(liczba_powtorzen/(liczba_powtorzen/10))):
        for j in range(math.ceil(liczba_powtorzen/10)):
            if(math.ceil(liczba_powtorzen/10)==1):
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                y_train_mean = y_train.copy()
                y_train_mean[outliers]=y_train.mean()
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
                   ax[0].set_title('Test z zastąpieniem danych odstających nr')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test z zastąpieniem danych odstających nr '+str(liczba_powtorzen))
                    break
                 
                   
            else:
                X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2, shuffle=True)
                outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
                y_train_mean = y_train.copy()
                y_train_mean[outliers]=y_train.mean()
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
                    ax[0].set_title('Test z zastąpieniem sanych odstających nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test z zastąpieniem danych odstających nr '+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1


def funkcja3(liczba_powtorzen):
    fig,ax =plt.subplots(2,1,figsize=(10,10)) #plt.subplots(int(liczba_powtorzen/(liczba_powtorzen/10)),math.ceil(liczba_powtorzen/10),figsize=(15,10))
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
                   ax[0].set_title('Test bez modyfikacji danych odstajacych nr 1')
                   break
                if ((i+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test bez modyfikacji danych odstajacych nr '+str(liczba_powtorzen))
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
                    ax[0].set_title('Test bez modyfikacji danych odstajacych nr 1')
                    break
                if((i+1)*(j+1)==liczba_powtorzen):
                    ax[1].scatter(y_test,y_pred)
                    ax[1].plot([minval,maxval],[minval,maxval])
                    ax[1].set_xlabel('y_test')
                    ax[1].set_ylabel('y_pred')
                    ax[1].set_title('Test bez modyfikacji danych odstajacych nr'+str(liczba_powtorzen))
                    break
                    
    plt.show()
    plt.tight_layout()           
    mape1 = mean_absolute_percentage_error (y_test,y_pred) 
    return mape1

print('Sredni procent bledu regresji po usunięciu danych odstających: ',funkcja1(1000)*100,'%')
print('Sredni procent bledu regresji po zastąpieniu danych odstających: ',funkcja2(1000)*100,'%')
print('Sredni procent bledu regresji bez modyfikowania danych odstających: ',funkcja3(1000)*100,'%')
