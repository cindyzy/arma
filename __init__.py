from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA, ARIMA
import statsmodels.tsa.stattools as ts
from sklearn import datasets, linear_model

def linear_line( X_parameters, Y_parameters):#线性回归
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    print("Intercept value ", predictions['intercept'],"coefficient" ,predictions['coefficient'] )
    return predictions
def armamodel(model):#ar模型
#
    arma_mod40=ARMA(model, order=(4,0)).fit(disp=-1)
    #print(arma_mod40.summary())
    print(arma_mod40.params)
    print(arma_mod40.conf_int())
    ts_predict=arma_mod40.predict('2018-1-1', '2018-8-22')
#    ts_predict, ax = plt.subplots(figsize=(10,8))
#    ts_predict = arma_mod40.plot_predict(start='2018-1-2 2:00:00', end='2018-1-18  18:00:00', ax=ax)
#    ts_predict, ax = plt.subplots(figsize=(12, 8))
#    ax = model.ix['2018':].plot(ax=ax)
#    ts_predict = arma_mod40.plot_predict('2018', '2019', dynamic=True, ax=ax, plot_insample=False)
    #print("_predict ", ts_predict)
    return ts_predict

data = pd.read_csv('./test.SS.csv', index_col='year')
data.index = pd.to_datetime(data.index)
X_parameter = []
Y_parameter = []
for single_hour ,single_like in zip(data['number'],data['like']):
    X_parameter.append([float(single_hour)])
    Y_parameter.append(float(single_like))


linear_line( X_parameter, Y_parameter)
regr = linear_model.LinearRegression()
regr.fit(X_parameter, Y_parameter)
dt=[]
dt=regr.predict(X_parameter)
et=data['like']-dt
print('线性', dt)
print('残差序列', et)

fc=np.array(et)
N=len(fc)
sum1=fc.sum()
#print(sum1)
mean=sum1/N
print("et均值", int(mean))
narray2=(fc-mean)*(fc-mean)
#print(narray2)
sum2=narray2.sum()
#print(sum2)
var=sum2/N
print("et的方差", var)
ts_predict=armamodel(et)
print('ts_predict', ts_predict)
#
#adftest = ts.adfuller(et, autolag='AIC')
#adf_res = pd.Series(adftest[0:1], index=['like'])
#for key, value in adftest[4].items():
#    adf_res['Critical Value (%s)' % key] = value

##print('like', int(adf_res['like']))
prex=[]
for number in range(1035, 1225, 5):
    X_parameter.append([float(number)])
##print ("i:", X_parameter)
prext=regr.predict(X_parameter)
pre_final=prext+ts_predict
print(pre_final)
##
#plt.clf()
##print("like", data['like'])

#plt.plot(ts_predict, label="PDT")
#plt.plot(dt, label = "ORG")
plt.plot(pre_final, label="PDT2")
plt.plot(data['like'], label = "ORG2")

plt.legend(loc="best")
plt.title("AR Test %s" % 4)
#plt.show()
#plt.scatter(X_parameter,Y_parameter,color='blue')
#plt.plot(X_parameter,regr.predict(X_parameter),color='red',linewidth=4)
#plt.xticks(())
#plt.yticks(())
plt.show()
print((0.2769-0.4564+0.3915-0.4787)*3.0277)
