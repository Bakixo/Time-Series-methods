""" 

ARIMA MODEL - STATİSTİCS MODELS

"""

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean() 
y = y.fillna(y.bfill())

train = y[:"1997-12-01"]
test = y["1998-01-01":]


def plot_Co2(train,test,y_pred,title):
    mae = mean_absolute_error(test,y_pred)
    train["1985":].plot(legend=True,label="TRAIN",title =f"{title}, MAE : {round(mae,2)}")
    test.plot(legend=True,label="TEST",figsize=(6,4))
    y_pred.plot(legend=True,label="PREDICTION")
    plt.show()

arimamodel = ARIMA(train, order=(1,1,1)).fit()
y_pred = arimamodel.forecast(48)[0:]
y_pred = pd.Series(y_pred,index=test.index)
#plot_Co2(train,test,y_pred,"ARIMA")



"""
#Hiper parametre optimizasyonu 

"""

#####################################

# AIC & BIC İstatistiklerine göre model derecesini belirleme

#####################################

p = d = q = range(0,2)
pdq=list(itertools.product(p,d,q))


def arima_optimizer(train,orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arima_model = ARIMA(train, order=order).fit()
            aic = arima_model.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%S AIC=%.2F' % (order,aic))
        except:
            continue
    print('BEST ARIMA%s AIC=%.2F' % (best_params,best_aic))
    return best_params

best_parametres = arima_optimizer(train,pdq)


###############
# final-model   
##############

final_arima_model = ARIMA(train, order=best_parametres).fit()
y_pred_final = final_arima_model.forecast(48)
y_pred_final = pd.Series(y_pred_final, index=test.index)


plot_Co2(train,test,y_pred_final,"Final ARIMA")









