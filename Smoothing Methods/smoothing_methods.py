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

warnings.filterwarnings('ignore')

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean() # resample = yeniden örnekle 'MS' ise haftalık hale getirme komutu
"""
bu satırı yazdıktan sonra bazı değerler NaN oluyor ve bu noktada
daha önceden alışık olunan makine öğrenimi metodlarıyla ortalamayla doldur vs şeklinde doldurma yapamıyoruz da
bir önceki veya bir sonraki değerlerle doldurulabilir veya onların ortalamasını alırsak olabilir ama tüm verilerin
ortalamasını alıp boş değerlere doldur demek zaman serilerinde uygun bir hamle değildir.

"""
eksik_değerler = y.isna().sum() # 5 adet
y = y.fillna(y.bfill()) # bir sonraki değerle doldurdu

y.plot(figsize=(15,6))
plt.show()

# ses yoluyla bir model kurulacak ama önemli : zaman serileri modeli kurarken makine öğrenmesi gibi testsplit vs yoktur bütün veri üzerinden eğitilir
# ancak ben yine de makine öğrenmesi mantığıyla ilerliycem ve veriyi ikiye bölücem. 


""" 
HOLDOUT İŞLEMİ 
"""

train = y[:'1997-12-01']
len(train) #478 ay 

test = y['1998-01-01':]
len(test) #48 ay




"""
ZAMAN SERİSİNİN YAPISAL ANALİZİ (durağanlık, trend, mevsimsellik analizi)

"""
#durağanlık testi teorik karşılığı (Dickey-Fuller Testi)

def is_stationary(y):

    # "H0: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p_value :{round(p_value,3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p_value :{round(p_value,3)})")



#Zaman serileri bileşenleri(level, trend, mevsimsellik ve residuals (artıklar)) ve durağanlık testi bir arada   


def ts_decompose(y, model="additive", stationary="True"):

    result = seasonal_decompose(y, model=model)

    fig, axes = plt.subplots(4,1,sharex=True,sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + "model")
    axes[0].plot(y,'k',label="original" + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend,label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(y,'k',label="original" + model)
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid,'r',label="Residuals & Mean" + str(round(result.resid.mean(),4)))
    axes[3].legend(loc='upper left')

    plt.show(block=True)

    if stationary:
        is_stationary(y)


"""
SİNGLE EXPONENTİAL SMOOTHİNG 

"""

# SES = Level

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

y_pred = ses_model.forecast(48)
mae = mean_absolute_error(test,y_pred)



def plot_Co2(train,test,y_pred,title):
    mae = mean_absolute_error(test,y_pred)
    train["1985":].plot(legend=True,label="TRAIN",title =f"{title}, MAE : {round(mae,2)}")
    test.plot(legend=True,label="TEST",figsize=(6,4))
    y_pred.plot(legend=True,label="PREDICTION")
    plt.show()




"""
HYPERPARAMETER OPTİMİZATİON

"""


# brute force işlemi 
def ses_optimizer(train, alphas, step = 48):
    best_alpha, best_mae = None, float("inf")
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test,y_pred)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
        print("Alpha: ",round(alpha,2), "mae : ",round (mae,4))
    print("Best Alpha is : ",round(best_alpha,2), "Best mae :", round(best_mae,4))
    return best_alpha,best_mae

alphas = np.arange(0.8, 1, 0.01)

best_alpha, best_mae = ses_optimizer(train,alphas)


"""

Final Model 

"""

final_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred= ses_model.forecast(48)

plot_Co2(train,test,y_pred,"SES MODEL SONUÇ")




##############################################################################################################





""" 
Double_Exponential Method

"""                                           #additive (toplamsal), Multiplicative (çarpımsal)
double_model = ExponentialSmoothing(train,trend="add").fit(smoothing_level=0.5,
                                                           smoothing_trend=0.5)

y_pred_double = double_model.forecast(48)

plot_Co2(train,test,y_pred_double,"Double_mode")


"""

Double_hyper_parametres

"""

# hiper parametreleri single daki gibi arama yapmıycaz çünkü sadece alpha yok beta da var bu yüzden bunların değişim kesişimini bulcaz.


def des_optimizer(train, alphas, betas, step = 48):
    best_alpha,best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            double_model = ExponentialSmoothing(train,trend="add").fit(smoothing_level=alpha, smoothing_trend=beta)
            y_pred = double_model.forecast(step)
            mae = mean_absolute_error(test,y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("Alpha: ",round(alpha,2), "Beta : ", round(beta,2), "mae : ",round (mae,4))
    print("Best Alpha is : ",round(best_alpha, 2), "Best beta :", round(best_beta, 2), "Best mae :", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01,1,0.10)
betas = np.arange(0.01,1,0.10)

best_alpha, best_beta, best_mae = des_optimizer(train,alphas,betas)


"""
Final_double_model

"""

final_double_model = ExponentialSmoothing(train,trend="add").fit(smoothing_level=best_alpha,smoothing_trend=best_beta)

final_y_pred = final_double_model.forecast(48)

plot_Co2(train,test,final_y_pred,"Double_mode")

#burada model çarpımsal olsaydı daha mı iyi çalışırdı falan bakmak isterseniz "add" yerine "mul" yazıp mae değerine göre yorum yapabilirsiniz.


"""

TRİPLE EXPONENTİAL METHOD 

"""

# TES = SES + DES + MEVSİMSELLİK

tes_model = ExponentialSmoothing(train,trend="add",seasonal="add",seasonal_periods=12).fit(smoothing_level=0.5,
                                                                                            smoothing_trend=0.5,
                                                                                            smoothing_seasonal=0.5)


tes_y_pred = tes_model.forecast(48)
plot_Co2(train,test,tes_y_pred,"tes model first")


"""
HYPER-OPTİMİZATİON FOR TES
"""

alpha = beta = gamma = np.arange(0.20, 1, 0.10)
abg = list(itertools.product(alpha,beta,gamma))
dfabg = pd.DataFrame(itertools.product(alpha,beta,gamma))

def tes_optimizer(train, abg, step = 48):
    best_alpha,best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model_gardas = ExponentialSmoothing(train,trend="add",seasonal="add",seasonal_periods=12).fit(smoothing_level=comb[0], smoothing_trend=comb[1], smoothing_seasonal=comb[2])
        tes_y_pred = tes_model_gardas.forecast(step)
        mae = mean_absolute_error(test,tes_y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0],2),round(comb[1],2), round(comb[2],2), round(mae,4)])
    print("Best Alpha is : ",round(best_alpha, 2), "Best beta :", round(best_beta, 2), "Best gamma is : ",round(best_gamma,2), "Best mae :", round(best_mae, 4))
    return best_alpha, best_beta, best_gamma, best_mae

best_ALPHA, best_BETA, best_GAMMA,best_MAE = tes_optimizer(train,abg)



son_tes_model = ExponentialSmoothing(train,trend="add",seasonal="add",seasonal_periods=12).fit(smoothing_level=best_ALPHA,
                                                                                            smoothing_trend=best_BETA,
                                                                                            smoothing_seasonal=best_GAMMA)

son_tes_pred = son_tes_model.forecast(48)
plot_Co2(train,test,son_tes_pred,"Son model")