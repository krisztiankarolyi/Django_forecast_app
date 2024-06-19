
import array
import copy
from tkinter import Image
import traceback
from typing import Any
from django.db import models
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import r2_score
import base64
import io
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy.stats import kstest
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from statsmodels.stats.diagnostic import het_white
import seaborn as sns
from AllamVizsga import settings

class Idosor :
    def __init__(self, idosor_nev:str, tanito_adatok: list, teszt_adatok: list, idoszakok: list, teszt_idoszakok: list):
        self.idosor_nev = idosor_nev
        self.tanito_adatok = tanito_adatok
        self.idoszakok = idoszakok
        self.teszt_adatok = teszt_adatok
        self.teszt_idoszakok = teszt_idoszakok
        self.adf = {}; self.kpss = {}
        self.Kolmogorov_Smirnov = {'statisztika': 0, 'p_value': 0}

    def calculateStatistics(self):
        self.tanitoDb = len(self.tanito_adatok)
        self.tesztDb = len(self.teszt_adatok)
        self.atlag = round(np.mean(self.tanito_adatok), 2)
        self.szoras = round(np.std(self.tanito_adatok), 2)
        self.variancia = round(np.var(self.tanito_adatok), 2)
        self.median = round(np.median(self.tanito_adatok), 2)
        self.min = np.min(self.tanito_adatok)
        self.max = np.max(self.tanito_adatok)
        self.minDatum = self.idoszakok[list.index(self.tanito_adatok, self.min)]
        self.maxDatum = self.idoszakok[list.index(self.tanito_adatok, self.max)]
        self.StationarityTest()
        self.distributionPlot = distributionPlot(data=self.tanito_adatok, name=self.idosor_nev, bins_=7)
        ks_statistic, p_value = kstest(self.tanito_adatok, 'norm')
        self.Kolmogorov_Smirnov['statisztika'] = ks_statistic
        self.Kolmogorov_Smirnov['p_value'] = p_value

    def StationarityTest(self, d:int=1, period: int = 1):
        adf_result = adfuller(self.tanito_adatok)
        kpss_result = kpss(self.tanito_adatok)

        self.adf["adf_stat"] = round(adf_result[0], 2)
        self.adf["p_value"] = round(adf_result[1], 2)
        self.adf["critical_values"] = {'5':0}
        self.adf["critical_values"]['5'] = round(adf_result[4]["5%"], 2)
        self.kpss["kpss_stat"] = round(kpss_result[0], 2)
        self.kpss["p_value"] = round(kpss_result[1], 2)
        self.kpss["critical_values"] = {'5':0}
        self.kpss["critical_values"]['5'] = round(kpss_result[3]["5%"], 2)

        differenced = np.diff(a=np.array(self.tanito_adatok), n=d)
        adf_result_diff = adfuller(differenced)
        kpss_result_diff= kpss(differenced)
        self.adf["diff_adf_stat"] = round(adf_result_diff[0], 2)
        self.adf["diff_p_value"] = round(adf_result_diff[1], 2)
        self.adf["diff_critical_values"] = {'5':0}
        self.adf["diff_critical_values"]['5'] = round(adf_result_diff[4]["5%"], 2)
        self.kpss["diff_kpss_stat"] = round(kpss_result_diff[0], 2)
        self.kpss["diff_p_value"] = round(kpss_result_diff[1], 2)
        self.kpss["diff_critical_values"] = {'5':0}
        self.kpss["diff_critical_values"]['5'] = round(kpss_result_diff[3]["5%"], 2)
    
    def autocorrelationPlot(self):
        buffer = io.BytesIO()
        fig, ax = plt.subplots()
        pd.plotting.autocorrelation_plot(self.tanito_adatok, ax=ax)
        plt.title(self.idosor_nev)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return encoded_image
       
    def plot_acf_and_pacf(self, x: list = [], pacf: bool = True):
        """Az idősor autokorrelációs és parc. autokorrelációs függvényeinek ábrázolása"""
        if len(x) == 0:
            x = self.tanito_adatok
        fig, (ax1, ax2) = plt.subplots(2 if pacf else 1, 1, figsize=(6, 6), sharex=False)
        fig.subplots_adjust(hspace=0.3)
        
        max_lags = len(x) // 2 if len(x) > 12 else len(x)
        lags = min(max_lags, 20)  # Set maximum lags to 20 or half the sample size, whichever is smaller
        
        plot_acf(x, lags=lags, ax=ax1, title=f"Autokorreláció ({self.idosor_nev})")
        
        if pacf:
            plot_pacf(x, lags=lags, ax=ax2, title=f"Parciális Autokorreláció ({self.idosor_nev})")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.pacf_acf_Diagram = encoded_image
        return encoded_image
          
    def predict_with_mlp(self, actFunction="relu", hidden_layers=(12, 12, 12), max_iters=3000, 
                         scalerMode="standard", randomStateMax=70, randomStateMin=50, solver="adam", 
                         targetMSE=0.1, n_delays = 3, n_pred=6, normOut: bool = False, stop: bool = False, learning_rate = "constant", alpha = 0.0001):
        """Az idősorhoz egy MLP regressziós modellt készít, amit betanít és elkészíti az előrejelzéseket. 
        Megoldja az adatok (de)normalizálását. A tanulás és általánosítás folyamatát kiértékeli. """
        if not self.teszt_adatok:
            print("Nincsenek tesztelési adatok.")
            return          
        
        # az adatok n darab korábbi megfigyeléstől függnek (késleltetett értékek)
        #adatok átcsoportosítása, hogy kijöjjön annyi jóslat, amennyit a test data alapból tartalamzott.
        test_data = self.tanito_adatok[-n_delays:]  + self.teszt_adatok
        learning_data = self.tanito_adatok[:-n_delays]
        self.x_scaler = self.y_scaler = None

        #a tanító és tesztelő adatbázis létrehozása
        self.x_train, self.y_train = split_sequence(learning_data, n_delays)
        self.x_test, self.y_test = split_sequence(test_data, n_delays)
        self.normalize(scalerMode, normOut=normOut)
        self.trainingPairs = zip(self.x_train, self.y_train)
        self.testingPairs = zip(self.x_test, self.y_test)

        self.mlp_model = MLP(actFunction=actFunction, hidden_layers=hidden_layers, max_iters=max_iters, random_state=randomStateMin, scaler=self.x_scaler, scalerMode=scalerMode, solver=solver, learning_rate = "constant", alpha = 0.0001)
        # a tanítás többszöri lefuttatása eltérő súlyozási kezdőértékekkel, és a legjobb kezdőérték megtartása. 
        # a legjobb kezdőérték az, amely mellett a modell a legjobban általánosít.
        self.random_state, n = self.find_best_random_state(randomStateMin, randomStateMax, targetMSE)
        self.mlp_model.random_state = self.random_state
        self.mlp_model.train_model(self.x_train, self.y_train, self.x_test, self.y_test, n, earlyStop=stop, solver=solver)
        self.mlp_model.recursiveForecast(n_pred, self.x_test, scaler_x=self.x_scaler, scaler_y=self.y_scaler, scalerMode=scalerMode, normOut=normOut)
        self.mlp_model.predictions = self.mlp_model.predict(self.x_test)
        self.mlp_model.predictions = np.array(self.mlp_model.predictions)
        self.deNormalize(scalerMode, normOut=normOut)
        self.mlp_model.calculateErrors()

        if(str.upper(solver) != "LBFGS"):
            self.mlp_model.lossCurves()
    
    def normalize(self, scalerMode: str, normOut: bool):
        """A tanító- és tesztelő adatok normalizálása, ha kért ilyet a felhasználó. 
        Szabadon választható, hogy az elvárt kimeneteket is normalizáljuk-e. """
        if (scalerMode == "robust"):
            self.x_scaler = RobustScaler()
            self.y_scaler = RobustScaler()
        if (scalerMode == "minmax"):
           self.x_scaler = MinMaxScaler()
           self.y_scaler = MinMaxScaler()
        if(scalerMode == "standard"):
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
           
        if scalerMode != "-":
            if(scalerMode == "log"):
                self.x_train = np.log(self.x_train)
                self.x_test = np.log(self.x_test)

                if(normOut):
                    self.y_test = np.log(self.y_test)
                    self.y_train = np.log(self.y_train)
                    
            else:
                self.x_train = self.x_scaler.fit_transform(np.array(self.x_train))
                self.x_test = self.x_scaler.transform(np.array(self.x_test))

                if(normOut):
                    self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1))
                    self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1))
                    self.y_train =[item for sublist in self.y_train for item in sublist]               
                    self.y_test = [item for sublist in self.y_test for item in sublist]
                else:
                    # az elorejelzesek visszatranszformalasara akkor is szukseg lesz a kimenetek visszaalakitasara
                    self.y_scaler.fit_transform(np.array(self.y_train).reshape(-1, 1))
                                           
    def deNormalize(self, scalerMode: str, normOut: bool):
          #Ha volt normalizálás, akkor a megjelenítés és hibaszámítás előtt vissza kell alakítani a becsléseket és a tanítóadatokat
          if scalerMode != "-":
            if(scalerMode == "log"):      
                self.mlp_model.x_test = np.exp(self.mlp_model.x_test)
                self.mlp_model.x_train = np.exp(self.mlp_model.x_train)    
                if(normOut):
                    self.mlp_model.learning_pred = np.exp(self.mlp_model.learning_pred)
                    self.mlp_model.y_train = np.exp(self.mlp_model.y_train )
                    self.mlp_model.y_test = np.exp(self.mlp_model.y_test)
                    self.mlp_model.predictions = np.exp(self.mlp_model.predictions)               
            else:
                self.mlp_model.x_train = self.x_scaler.inverse_transform(self.mlp_model.x_train)
                self.mlp_model.x_test = self.x_scaler.inverse_transform(self.mlp_model.x_test)

                if(normOut):
                    self.mlp_model.y_train = self.y_scaler.inverse_transform(np.array(self.mlp_model.y_train).reshape(-1, 1)).tolist()
                    self.mlp_model.y_test = self.y_scaler.inverse_transform(np.array(self.mlp_model.y_test).reshape(-1, 1)).tolist()

                    predictions = self.y_scaler.inverse_transform(np.array(self.mlp_model.predictions).reshape(-1,1)).tolist()
                    learning_pred = self.y_scaler.inverse_transform(np.array(self.mlp_model.learning_pred).reshape(-1,1)).tolist() 

                    self.mlp_model.y_train = [item for sublist in self.mlp_model.y_train for item in sublist]
                    self.mlp_model.y_test = [item for sublist in self.mlp_model.y_test for item in sublist]
                    
                    self.mlp_model.learning_pred = [item for sublist in learning_pred for item in sublist]
                    self.mlp_model.predictions = [item for sublist in predictions for item in sublist]

    def get_month_number(self, month):
        months = {
            'január': 1,
            'február': 2,
            'március': 3,
            'április': 4,
            'május': 5,
            'június': 6,
            'július': 7,
            'augusztus': 8,
            'szeptember': 9,
            'október': 10,
            'november': 11,
            'december': 12
        }
        return months[month]

    def find_best_random_state(self, random_state_min=50, random_state_max=70, targetMSE=0):
        """Fits the model with different random starting value for weights,
          and keeps the best value where the model performed the best on the test set based on MSE. If the target MSE is already reached, the iteration stops. """
        best_random_state = 0
        best_mse = float(1000) 
        model = copy.deepcopy(self.mlp_model.model)

        for random_state in range(random_state_min, random_state_max+1):
            model = model.fit(self.x_train, self.y_train)
            model.random_state = random_state
            predictions = model.predict(self.x_test)
            mse = mean_squared_error(self.y_test, predictions)
            print(f"trying {self.idosor_nev}'s MLP prediction with random state {random_state} --> MSE: {mse}")

            if mse < best_mse:
                best_mse = mse
                best_random_state = random_state
            
            if round(mse, 2) <= targetMSE:
                print(f"target RRMSE{targetMSE} reached, stopping search...")
                return best_random_state, best_mse

        self.random_state = best_random_state
        try:
            n_iter = model.n_iter_
        except:
            n_iter = 0

        return best_random_state, n_iter

    def predictARIMA(self, p:int = 1, d: int = 0, q: int = 0, n_pred:int = 6, autoArima: bool = False):
        """ARIMA modellek készítése, tanítása és előrejelzés n_pred értékig."""
        t = len(self.teszt_adatok)
        self.ARIMA = ARIMA(p, d, q, tanito_adatok=self.tanito_adatok, teszt_adatok=self.teszt_adatok, idoszakok=self.idoszakok, teszt_idoszakok=self.teszt_idoszakok, n_pred = n_pred, autoArima=autoArima, modelName=self.idosor_nev)
        model_fit = self.ARIMA.fit(self.tanito_adatok)
        #a tanitoadatokra való illeszkedés, es a legjobb illesztesi becslesek lekerese.
        #a diff. soran az elso d db becsles elveszik, vagyis 0 lesz, ami torzitana a reziduumok kepet, igy ezeket lehagyjuk
   
        #hibamutatok es reziduumok vizsgalata az illesztesre
        self.ARIMA.EvaluateTrainingProcess()
        #becslések a teszt időszakra
        self.ARIMA.becslesek = self.ARIMA.rollingForecast(train_data=self.tanito_adatok, test_data=self.teszt_adatok, mode="test")
        #az utolso n_pred elem mar a tanitoadaton tuli elorejelzesek
        self.ARIMA.elorejelzesek = self.ARIMA.rollingForecast(train_data=self.tanito_adatok, test_data=self.teszt_adatok, mode="forecast", n_pred=n_pred)
        self.ARIMA.forecast_diagram = Diagram(self.ARIMA.elorejelzesek, showValues=True)
        self.ARIMA.train_diagram = Diagram(self.ARIMA.fitted_values, self.tanito_adatok[int(d):], x=self.idoszakok[int(d):], label1="Becslések", label2="valódi adatok")
        self.ARIMA.test_diagram = Diagram(self.ARIMA.becslesek, self.teszt_adatok,x=self.teszt_idoszakok, label1="Becslések", label2="valódi adatok")
        self.ARIMA.EvaluateTestProcess()
                 
        return self.ARIMA

def Diagram(y:list, y2:list=[], x:list=[], label1:str="Train", label2:str="Forecast", showValues:bool=False):
        """1 vagy 2 idősort ábrázol, base64 kódolásban adja vissza a képet. 
        Példa használat HTML-ben: <img> src='data:image/png;base64,{{ base64_image }}</img>'"""
        plt.figure(figsize=(16, 9))
        plt.plot(y, label=label1)
        if len(y2) > 0:
            plt.plot(y2, label=label2)
        # Kiegészítjük az x tengely értékeit, ha az egyik sorozat hosszabb, mint a másik
        max_len = max(len(y), len(y2))
        if not x or len(x) != len(y):
            x = list(range(1, max_len + 1))

        # Lépésköz dinamikius meghatározása  alapján
        step = max(1, int(len(x) / 12))
        plt.xticks(range(0, max_len, step), x[::step], rotation=90)  # x tengely feliratok beállítása
        if len(y2) > 0:
            plt.ylim(min(min(y), min(y2))-1, max(max(y), max(y2))+1)
        else:
                plt.ylim(min(y)-1, max(y)+1)
        
        if showValues:
            for i, value in enumerate(y):
                 plt.text(i, value, f'{value:.2f}', ha='right', va='bottom')
            if len(y2) > 0:
                for i, value in enumerate(y2):
                    plt.text(i, value, f'{value:.2f}', ha='right', va='top')
                
        plt.legend()
        plt.grid(True, 'major', 'both')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
def split_sequence(sequence, n_steps):
    """idősorból gépi tanulási tanító adatbázis készítése. 
        Visszatérés: x (inputok listája) és egy y (targetek listája)
    n_steps db inputból következik egy kimenet. 
    Pl. n_steps = 3: 
    x_train=[[1,2,3], [4,5,6], ...]  
    és y_train = [4, 7]"""
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

class ARIMA:
    def __init__(self, p:int = 1, d: int = 0, q: int = 0, tanito_adatok = [], 
                teszt_adatok = [], idoszakok = [], teszt_idoszakok = [], n_pred: int = 6,
                autoArima: bool = False, modelName=""):
        
        self.becslesek = self.fitted_values = self.forecasts = [], 
        self.aic = self.bic = self.mseTrain = self.rrmseTrain = self.mapeTrain = self.r2Train = 0
        self.mseTest = self.rrmseTest = self.mapeTest = self.r2Test = 0
        self.test_diagram = self.train_diagram = self.forecast_diagram= None
        self.modelName = modelName
        self.n = n_pred
        self.errorMatrix = None
        self.errorHistogram = None
        self.tanito_adatok = tanito_adatok
        self.teszt_adatok = teszt_adatok
        self.teszt_idoszakok = teszt_idoszakok
        self.coefs = {}

        if(autoArima):
            self.autoARIMA()
        else:
            self.p = int(p)
            self.q = int(q)
            self.d = int(d)
        self.modelName += f" ARIMA ({self.p}, {self.d}, {self.q})"

        if tanito_adatok is not None and teszt_adatok is not None and idoszakok is not None and teszt_idoszakok is not None:
            if len(tanito_adatok) == len(idoszakok) and len(teszt_adatok) == len(teszt_idoszakok):
                self.t = len(teszt_adatok)   
        else:
            print("Nem megfelelő adatstruktúra megadva.")
    
    def autoARIMA(self):
        """Többféle modell kipróbálása, és a legjobb modell (AIC alapján) megtartása"""
        from pmdarima import auto_arima
        import pandas as pd
        model = auto_arima(y = self.tanito_adatok, start_p=1, start_q=0, max_p=5, max_q=5, min_d = 1, 
                           max_d = 3, seasonal = False,  trace=True,error_action='ignore',stationary=False, 
                           suppress_warnings=True, stepwise=True, )
        print(model.summary())
        best_order = model.order
        self.p = int(best_order[0])
        self.d = int(best_order[1])
        self.q = int(best_order[2])
    
    def EvaluateTrainingProcess(self):
        try:
            d = int(self.d)
            self.aic = self.model_fit.aic
            self.bic = self.model_fit.bic
            self.mseTrain = MSE(self.tanito_adatok[d:], self.fitted_values)
            self.rrmseTrain = RRMSE(self.tanito_adatok[d:], self.fitted_values)
            self.mapeTrain = MAPE(self.tanito_adatok[d:], self.fitted_values)
            self.r2Train = R2(self.tanito_adatok[d:], self.fitted_values)

            if (len(self.fitted_values != len(self.tanito_adatok[d:]))):
                becslesek, adatok = align_lengths(self.fitted_values, self.tanito_adatok[d:])

            self.TrainResiduals = np.array([adatok[i] - becslesek[i] for i in range(len(adatok))])[1:]
            self.TrainErrorHistogram = error_distribution_plot(residuals=self.TrainResiduals, name="")
            self.TrainResidualsPlot = plot_Residuals(residuals=self.TrainResiduals, name="")
            self.TrainWhite = White(self.TrainResiduals)
            self.TrainResACFPlot = acfPlot(self.TrainResiduals)

            self.coefs["AR"] = self.model_fit.arparams
            self.coefs["MA"] = self.model_fit.maparams       

        except Exception as exp:
            print("Hiba történt")
            print(traceback.format_exc(exp))

    def EvaluateTestProcess(self):   
        self.mseTest = MSE(self.teszt_adatok, self.becslesek)
        self.rrmseTest = RRMSE(self.teszt_adatok, self.becslesek)
        self.mapeTest = MAPE(self.teszt_adatok, self.becslesek)
        self.r2Test = R2(self.teszt_adatok, self.becslesek)

        try:
            self.Testresiduals = np.array([self.teszt_adatok[i] - self.becslesek[i] for i in range(len(self.teszt_adatok))])
            self.TesterrorHistogram = error_distribution_plot(residuals=self.Testresiduals, name="")
            self.TestresidualsPlot = plot_Residuals(residuals=self.Testresiduals)
            self.Testwhite = White(self.Testresiduals)
            self.TestresACFPlot = acfPlot(self.Testresiduals)
            self.becsleseksZipped  = zip(self.becslesek, self.teszt_adatok, self.Testresiduals)
            
        except Exception as exp:
            print("Hiba történt")
            print(traceback.format_exc(exp))

    def fit(self, adatok):
        # ARIMA modell illesztése
        self.model = sm.tsa.ARIMA(adatok, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()
        self.fitted_values =  self.model_fit.predict(start=int(self.d), end=len(self.tanito_adatok))

        return self.model_fit
    
    def rollingForecast(self, train_data:list, test_data: list, mode: str = "test", n_pred: int = 6):
        predictions = []
        if mode == "test": 
            history = list(test_data)
            n_pred = len(history)
        elif mode == "forecast":
            n_pred = n_pred
            history = list(train_data + test_data)
    
        for t in range(n_pred):
            model = sm.tsa.ARIMA(history, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            output = model_fit.forecast()
            prediction = output[0]
            predictions.append(prediction)

            if mode == "test":
                history.append(test_data[t])
            elif mode == "forecast":
                history.append(prediction)         
            
        return predictions
    

class MLP:
    def __init__(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=1000, 
                 random_state=50, scaler=None, scalerMode="-", solver="adam", learning_rate = "constant", alpha = 0.0001) :
        self.hidden_layers = hidden_layers
        self.NrofHiddenLayers = len(hidden_layers)
        self.max_iters = max_iters
        self.random_state = random_state
        self.activation = actFunction
        self.solver = solver
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, solver=solver,  activation=actFunction, max_iter=max_iters, random_state=random_state, n_iter_no_change=20, learning_rate=learning_rate, alpha=alpha)
        self.scaler = scaler
        self.y_scaler = None
        self.scalerMode = scalerMode

        self.mape_test = self.rrmse_test = self.mse_test = self.mape_train = self.rrmse_train = self.mse_train =  self.r2_train = self.r2_test = self.accuracy = 0
        self.diagram = self.trainDiagram = None
        self.modelStr = self.NrofHiddenLayers * '{}, '
        self.modelStr = "("+self.modelStr.format(*hidden_layers)[:-1]+")"
        self.x_test = self.y_test = self.x_train = self.y_train = self.predictions = self.weights = self.learning_pred = []

    def train_model(self, x_train: list, y_train: list, x_test: list, y_test: list, n: int,  earlyStop: bool, solver: str):
        """If solver is ADAM or SGD, it fits the model step-by-step and calculates loss curves.
          In case of LBFGS it uses the autamitc fit() function and the curves won't be calculated since its not possible"""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.trainLossCurve = []
        self.testLossCurve = []

        if str.upper(solver) == "ADAM" or str.upper(solver)  == "SGD":
            n = max(n, 200)
            if(not earlyStop):
                n = self.max_iters
            
            for i in range(int(n)):
                self.model = self.model.partial_fit(x_train, y_train)
                predictions = self.model.predict(x_train)
                train_loss = self.model.loss_
                self.trainLossCurve.append(train_loss)
                predictions = self.model.predict(self.x_test)
                test_loss = MSE(predictions, self.y_test)
                self.testLossCurve.append(test_loss)
        
        else:
            self.model = self.model.fit(x_train, y_train)

        self.weights = [layer_weights for layer_weights in self.model.coefs_]
        learning_pred = self.model.predict(x_train)
        self.learning_pred = learning_pred


    def lossCurves(self):
        """Veszteség görbék ábrázolása SGD és ADAM esetében"""
        plt.plot(self.trainLossCurve, label='Train')
        if(self.testLossCurve != []):
            plt.plot(self.testLossCurve, label='Test')
        max_loss_value = max(max(self.trainLossCurve), max(self.testLossCurve))
        if(str.upper(self.solver) == "LBFGS"):
            plt.ylim(-0.01, 0.02)
        else:
            plt.ylim(-0.25, max_loss_value ** (1/2))
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

        # Save the plot to a buffer in PNG format
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        self.lossCurve = base64.b64encode(buffer.read()).decode('utf-8')

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def recursiveForecast(self, n, x_test, scaler_x=None, scaler_y=None, scalerMode="-", normOut = False):
        """The model forecasts one value from the last known input, then the forecasted value will be used as input in the next step"""
        x_combined = np.concatenate((self.x_train, self.x_test))
        y_combined = np.concatenate((self.y_train, self.y_test))
        self.model.fit(x_combined, y_combined)

        forecasts = []
        x_axis = []    
        input = x_test[-1].reshape(1, -1)  # Átalakítjuk a legutolsó  input értéket 2D formátumra

        for i in range(n):
            forecast = self.predict(input)[0]  # a predict 2d-s lisát ad vissza, 1 elemmel, mert csak 1 input van
            print(f"foreacast before: {forecast}")
                
            if(normOut == False):
                if(scalerMode == "log"):
                    forecast = np.log(np.longdouble(forecast))

            forecasts.append(forecast) 
            print(f"{i+1}. : {input} ---> {forecast}")
            x_axis.append(f"{i+1}. jóslat")

            input = np.hstack((input[:, 1:], forecast.reshape(1, -1)))

            if scaler_y is not None and not normOut:
                print(f"input before: {input}")
               # print("A kimenetek ugyan nem voltak normalizálva, így a modell nem a skálába illő kimeneteket ad. Ezeket vissza kell alakítani, mielőtt felhasználja bemenetként")
                input[0][-1] = scaler_y.transform(np.array(input[0][-1]).reshape(-1, 1)) 
                print(f"input after: {input}")

        if(scalerMode == "log"):
            forecasts = np.exp(forecasts).tolist()

        elif(scaler_y is not None and normOut) :
            forecasts = scaler_y.inverse_transform(np.array(forecasts).reshape(1, -1)).flatten().tolist()
        

        self.forecastsPlot = Diagram(y=forecasts, label1="MLP előrejelzések", showValues=True)
        return forecasts

    def calculateErrors(self):
        self.mse_test = MSE(self.y_test, self.predictions)
        self.rrmse_test = RRMSE(self.y_test, self.predictions)
        self.rmse_test = RMSE(self.y_test, self.predictions)
        self.r2_test = r2_score(self.y_test, self.predictions)
        self.mape_test = MAPE(self.y_test, self.predictions)

        self.r2_train = R2(self.learning_pred, self.y_train)
        self.mse_train = MSE(self.learning_pred, self.y_train)
        self.rrmse_train = RRMSE(self.learning_pred, self.y_train)
        self.mape_train = MAPE(self.learning_pred, self.y_train)

        self.train_residuals =  np.array([self.y_train[i] - self.learning_pred[i] for i in range(len(self.y_train))])
        self.train_white = White(self.train_residuals)
        self.train_residualsPlot = plot_Residuals(residuals=self.train_residuals, name=self.modelStr)
        self.train_errorHistogram = error_distribution_plot(residuals=self.train_residuals, name=self.modelStr)
        self.train_resACFPlot = acfPlot(self.train_residuals)

        self.residuals =  np.array([self.y_test[i] - self.predictions[i] for i in range(len(self.y_test))])
        self.white = White(self.residuals)
        self.residualsPlot = plot_Residuals(residuals=self.residuals, name=self.modelStr)
        self.errorHistogram = error_distribution_plot(residuals=self.residuals, name=self.modelStr)
        self.resACFPlot = acfPlot(self.residuals)
        self.ResultsZipped = zip(self.predictions, self.y_test, self.residuals)  

        self.diagram = Diagram(self.predictions, self.y_test, "Becsült", "Valódi adat")
        self.trainDiagram = Diagram(self.learning_pred, self.y_train, "Becsült", "Valódi adat")

    
def align_lengths(list1, list2):
    """It makes the two lists' length equal"""
    min_length = min(len(list1), len(list2))
    return list1[:min_length], list2[:min_length]

def MSE(becslesek, teszt_adatok):
    try:
        becslesek, teszt_adatok = align_lengths(becslesek, teszt_adatok)
        n = len(teszt_adatok)
        teszt_adatok_np = np.array(teszt_adatok)
        becslesek_np = np.array(becslesek)
        mse = np.sum((teszt_adatok_np - becslesek_np)**2) / n
        return mse
    except Exception as e:
        print(traceback.format_exc())
        return -1   

def R2(becslesek, teszt_adatok):
    try:
        becslesek, teszt_adatok = align_lengths(becslesek, teszt_adatok)
        teszt_adatok_np = np.array(teszt_adatok)
        becslesek_np = np.array(becslesek)
        
        ss_res = np.sum((teszt_adatok_np - becslesek_np) ** 2)
        ss_tot = np.sum((teszt_adatok_np - np.mean(teszt_adatok_np)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    except Exception as e:
        print(traceback.format_exc())
        return -1
    
def RRMSE(becslesek, teszt_adatok):
    try:
        becslesek, teszt_adatok = align_lengths(becslesek, teszt_adatok)
        mse = MSE(becslesek, teszt_adatok)
        mean_y = np.mean(teszt_adatok)
        if mse < 0 or mean_y <= 0:
            rrmse = np.sqrt(-1 * mse) / mean_y
        else:  
            rrmse = np.sqrt(mse) / mean_y
        return rrmse
    except Exception as e:
        print(traceback.format_exc())
        return -1
    
def MAPE(becslesek, teszt_adatok):
    try:
        becslesek, teszt_adatok = align_lengths(becslesek, teszt_adatok)
        absolute_percentage_errors = []
        for prediction, actual in zip(becslesek, teszt_adatok):
            if actual == 0:
                continue
            absolute_percentage_error = abs((actual - prediction) / actual) * 100
            absolute_percentage_errors.append(absolute_percentage_error)
        if len(absolute_percentage_errors) == 0:
            return -1
        mean_absolute_percentage_error = sum(absolute_percentage_errors) / len(absolute_percentage_errors)
        return mean_absolute_percentage_error
    except Exception as e:
        print(traceback.format_exc())
        return -1

def RMSE(becslesek, teszt_adatok):
    try:
        becslesek, teszt_adatok = align_lengths(becslesek, teszt_adatok)
        mse = MSE(becslesek, teszt_adatok)
        rmse = np.sqrt(mse)
        return rmse
    except Exception as e:
        print(traceback.format_exc())
        return -1


import io
import base64
import numpy as np
import matplotlib.pyplot as plt

def error_distribution_plot(residuals, num_bins=10, name=""):
    # Minimum and maximum error values
    min_error, max_error = min(residuals), max(residuals)
    num_bins = int((len(residuals)) * 0.5)
    # Error histogram buffer
    hist_buffer = io.BytesIO()
    # Calculate histogram
    hist_values, bin_edges = np.histogram(residuals, bins=np.linspace(min_error, max_error, num_bins + 1))
    # Plot histogram
    plt.bar(bin_edges[:-1], hist_values, color='blue', edgecolor='black', width=bin_edges[1] - bin_edges[0])

    plt.xlabel('Tartomány')
    plt.ylabel('Gyakoriság')
    plt.title(name+' előrejelzési hibák eloszlása')
    # Set y-axis ticks to integer values
    plt.yticks(np.arange(0, max(hist_values) + 1, 1))

    plt.savefig(hist_buffer, format="png")
    hist_buffer.seek(0)
    encoded_hist_image = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
    plt.close()

    return encoded_hist_image

def plot_Residuals(residuals, name=""):
    """"""
    # Lineáris illesztés
    x = np.arange(len(residuals))
    slope, intercept = np.polyfit(x, residuals, 1)
    line = slope * x + intercept
    # Reziduumok grafikon buffer
    residuals_buffer = io.BytesIO()
    # Reziduumok grafikon
    plt.plot(residuals, marker='o', linestyle='', color='blue')
    plt.plot(line, linestyle='-', color='red')
    plt.xlabel('Előrejelzés sorszáma')
    plt.ylabel('Reziduum')
    plt.title(name+' Előrejelzések reziduumai')
    # Y tengely intervallum beállítása
    plt.ylim(np.min(residuals) - 2, np.max(residuals) + 2)
    plt.savefig(residuals_buffer, format="png")
    residuals_buffer.seek(0)
    encoded_residuals_plot = base64.b64encode(residuals_buffer.getvalue()).decode('utf-8')
    plt.close()
    return encoded_residuals_plot

def Ljung_Box(residuals):
    lag = len(residuals) // 4
    result = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=lag, return_df=True)
    p_values = [] 
    stats = []
    for i in range(lag):
        p_values.append(result.loc[i+1, 'lb_pvalue'])
        stats.append(result.loc[i+1, 'lb_stat'] )

    return stats, p_values

def White(residuals):
    """ Azt nézi, hogy a hibák varrianciája állandó-e azáltal, hogy homoszkedaszicitás vagy heteroszkedaszicitás van jelen.
        H0: Nincs Heteroszkedaszicitás (homoszkedaszicitás) ---> ez a jó, mert állandó  hibaszórás
        H1: Heteroscedasticity is present. --> nem jó
        Ha p > 0.05 nem utasítjuk el a nullhipotézist, tehát nincs jelen heteroszkedaszicitás, --> ez a jó
        Ha p < 0.05 akkor sajnos elutasítjuk H0-t, tehát a hibák varrianciája nem állandó"""
    squared_errors = np.square(residuals)
    exog = np.arange(len(squared_errors))
    exog = sm.add_constant(exog) 
    white_results = het_white(squared_errors, exog)

    p_value_homoskedasticity = white_results[1]
    p_value_heteroskedasticity = white_results[1]

    print(f"P-érték a homoszkedaszticitás teszthez: {p_value_homoskedasticity:.4f}")
    print(f"P-érték a heteroszkedaszticitás teszthez: {p_value_heteroskedasticity:.4f}")

    if p_value_heteroskedasticity < 0.05:
        res = f"White-teszt: <br> p = {round(p_value_heteroskedasticity, 2)} < 0.05  --> elutasítjuk H0-t. <br> A modell heteroszkedaszticitást mutat,  <br> tehát nem állandó a a hibák varrianciája, nem igazán megbízható."
    else:
        res = f"White-teszt: <br>  p = {round(p_value_heteroskedasticity,2)} > 0.05 --> nem utasítjuk H0-t. <br> A modell nem mutat heteroszkedaszticitást, <br> állandó a hibák varrianciája, megbízhatónak mondható."

    return res

def acfPlot(x: list = []):
    """
    returns the plot of autocorrelation function of the given series
    """
    x = np.array(x)
    lags = len(x) // 2 if len(x) > 12 else len(x)-1
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_acf(x, lags=lags, ax=ax)
    ax.set_title("Reziduumok autokorrelációi")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image

def distributionPlot(data, method="freedman", bins_ = 0, name=""):
    """Creates a histogram of the given data, and returns it as a base64 encoded PNG image"""
    adatok = data
    if(bins_ == 0): 
        if method == "freedman":
            IQR = np.percentile(adatok, 75) - np.percentile(adatok, 25)
            n = len(adatok)
            bins = int((max(adatok) - min(adatok)) / (2 * IQR * n**(-1/3)))
        elif method == "scott":
            std = np.std(adatok)
            n = len(adatok)
            bins = int(3.5 * std * n**(-1/3))
        else:
            bins = int(max(adatok) - min(adatok))
    else: bins = bins_

    n, bins, _ = plt.hist(data, bins=bins, color='blue', edgecolor='black')
    plt.xlabel('Értékek')
    plt.ylabel('Gyakoriság')
    plt.title(f"{name} Hisztogram")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    # Az x tengely feliratainak beállítása a létrehozott csoportokra
    plt.xticks(bins)

    plt.plot(bins[:-1], n, color='red', marker='o', linestyle='-', linewidth=2, markersize=6)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image
