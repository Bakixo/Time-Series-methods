"""

Sarima model - STATİSTİCS MODELS  

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
y = y.fillna(y.bfill())  # NaN değerlerini bfill ile doldurma

train = y[:"1997-12-01"]
test = y["1998-01-01":]


def plot_Co2(train, test, y_pred, title):
    # NaN kontrolü ve doldurma
    if test.isnull().sum() > 0:
        test = test.fillna(method='bfill').fillna(method='ffill')

    if y_pred.isnull().sum() > 0:
        y_pred = y_pred.fillna(method='bfill').fillna(method='ffill')

    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE : {round(mae, 2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()



model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))

sarima_model = model.fit()
sar_y_pred = sarima_model.get_forecast(steps=48)
pred = sar_y_pred.predicted_mean
pred = pd.Series(pred, index=test.index)

#plot_Co2(train, test, pred, "Final SARIMA")



"""

hyper-parametres optimization

"""


p = d = q = range(0,2)
pdq=list(itertools.product(p,d,q))
seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]


def sarima_optimizer(train, pdq, seasonal_pdq,display = False):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarima_model = SARIMAX(train, order=param, seasonal_order=param_seasonal).fit(display=False)
                aic = sarima_model.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC : {}'.format(param, param_seasonal, aic))
            except:
                continue
    print('Best SARIMA{}x{}12 - AIC : {}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order


best_order, best_seasonal_order = sarima_optimizer(train,pdq,seasonal_pdq,disp=True)

"""

final_Sarima model

"""

final_Sarima_model = SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order).fit()
y_pred_final = final_Sarima_model.forecast(48)
y_pred_final = pd.Series(y_pred_final, index=test.index)

#plot_Co2(train,test,y_pred_final,"FİNAL SARİMAX")


"""

MAE YE GÖRE OPTİMİZASYON 

"""

def sarima_optimizer(train, pdq, seasonal_pdq,disp=False):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarima_model = SARIMAX(train, order=param, seasonal_order=param_seasonal).fit(disp=False)
                y_pred = sarima_model.get_forecast(48)
                y_pred2 = y_pred.predicted_mean
                mae = mean_absolute_error(test,y_pred2)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - Mae : {}'.format(param, param_seasonal, mae))
            except:
                continue
    print('Best SARIMA{}x{}12 - Mae : {}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer(train,pdq,seasonal_pdq,disp=True)
yeter_model = SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order).fit()
yeter_pred = yeter_model.get_forecast(48)
yeter_pred2 = yeter_pred.predicted_mean
yeter_pred2 = pd.Series(yeter_pred2, index=test.index)

#plot_Co2(train,test,yeter_pred2,"YETER SARIMAX")



"""

MODELİ TEST EDELİM

"""
son_deger = y.tail(1)


finish_model = SARIMAX(y,order=best_order,seasonal_order=best_seasonal_order).fit()

feature_predict = finish_model.get_forecast(1)
feature_predict = feature_predict.predicted_mean

print(f"son_deger :{son_deger} feature_predict : {feature_predict}")

# 2001-12-01 tarihi için atmosferdeki karbondioksit yoğunluğunun 371.02 parça/milyon (ppm) olacağını tahmin ettik.
