import base64
from datetime import datetime
import os
from random import Random
import traceback
import matplotlib
import matplotlib.dates as mdates
from django.http import HttpResponse
from django.template import loader
import io
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from .models import Idosor
import statsmodels.api as sm
from django.contrib import messages
from django.shortcuts import redirect
from pandas.plotting import autocorrelation_plot

global idosorok 
idosorok = {}

def upload(request):
    """A feltöltő oldalt (kezdőlap) visszaadó fgv"""
    if not request.session.session_key:
        request.session.save()

    messages.error(request, 'Nem lett munadatforrás fájl feltöltve!')
    return render(request, 'upload.html')

def home(request):
    """A beolvasást, feldolgozást végző függvény, amely visszaadja az alapvető elemzések oldalát"""
    global idosorok

    if 'file' not in request.FILES or 'sheet' not in request.POST:
        messages.error(request, 'Hiányzó fájl vagy paraméterek! (munkalap nevek)!')
        return redirect('upload')

    uploaded_file = request.FILES['file']
    sheetName = request.POST['sheet']
    train_size = int(request.POST['train_size'])
    adatsorok, adatsorNevek, idoPontok, teszt_adatok = [], [], [], None

    try:
        global idosorok

        df = pd.read_excel(uploaded_file, sheet_name=sheetName)
        fejlec = df.columns.tolist()
        idoPontok = df[fejlec[0]].tolist()

        for i, col in enumerate(fejlec[1:]):
            adatsorNevek.append(col)
            adatsorok.append(df[col].tolist())
        
        idosorok[request.session.session_key] = createObjects(adatsorNevek, adatsorok, idoPontok, train_size)
        
        diagram = AbrazolEgyben(adatsorok=adatsorok, idoszakok=idoPontok, megnevezesek=adatsorNevek,  grid=True, num=4)
        diagram = base64.b64encode(diagram.read()).decode('utf-8')

        data_rows = [{'idoPont': ido, 'adatsorok': [adatsor[i] for adatsor in adatsorok]} for i, ido in enumerate(idoPontok)]
        
        for i in idosorok[request.session.session_key]:
            i.calculateStatistics()

        return render(request, 'home.html', {'data_rows': data_rows, 'adatsorNevek': adatsorNevek, 'idosorok': idosorok[request.session.session_key], 'diagram': diagram})
    
    except Exception:
        print(traceback.format_exc())
        messages.error(request, 'Nem található a munkalap!')
        return redirect('upload')

def BoxJenkins(request):
    """Az ARIMA modellek létrehozását és tesztelését lehetővé adó oldal visszaadása"""
    global idosorok
    for idosor in idosorok[request.session.session_key]:
        try:
            idosor.plot_acf_and_pacf()
        except Exception as e:
            print(e)
        
    return render(request, 'Box-Jenkins.html', {'idosorok': idosorok[request.session.session_key]})

def MLP(request):
    """Az MLP modellek létrehozását és tesztelését lehetővé adó oldal visszaadása"""
    global idosorok
    try: 
        return render(request, 'MLP.html', {'idosorok': idosorok[request.session.session_key]})
    except Exception as e:
        print(traceback.format_exc())
        return HttpResponse("Hiba történt. "+str(e))
    
    
def MLPResults(request):
    """Az MLP modellek paraméterezési oldalán levő űrlap beküldését követően 
    létrejönnek az idősor(ok)hoz rendelt regressziós modellek, azok be lesznek tanítva a paraméterek szerint
    és létrejönnek a tesztadatokra adott becslések + a kért előrejelzések.
    A tanulási és általánosítási folyamatot kiértékelő mutatók, táblázatok és grafikonokból létrejön a visszaküldendő weblap"""
    global idosorok
    try:
        n_pred =  int(request.POST["n_pred"])
        for idosor in idosorok[request.session.session_key]:           
            scaler =  request.POST[idosor.idosor_nev+'_scaler']
            actFunction = request.POST[idosor.idosor_nev+'_actFunction']
            maxIters = request.POST[idosor.idosor_nev+'_max_iters']
            targetRRMSE = float(request.POST[idosor.idosor_nev+'_targetRRMSE'])/100
            solver = request.POST[idosor.idosor_nev+"_solver"]
            n_delays = int(request.POST[idosor.idosor_nev+"_n_delay"])
            randomStateMin = int(request.POST[idosor.idosor_nev+'_random_state_min'])
            randomStateMax = int(request.POST[idosor.idosor_nev+'_random_state_max'])
            learning_rate = request.POST[idosor.idosor_nev+'_learning_rate']
            hidden_layers = tuple(map(int, request.POST[idosor.idosor_nev+'_hidden_layers'].split(',')))
            alpha = float(request.POST[idosor.idosor_nev+'_alpha'])
            normOut = False
            stop = False

            if idosor.idosor_nev+'_normOut' in request.POST:
                normOut = True
            if idosor.idosor_nev+'_stop' in request.POST:
                stop = True

            idosor.predict_with_mlp(actFunction=actFunction, hidden_layers=hidden_layers, max_iters= int(maxIters),
                                    scalerMode=scaler, randomStateMax=randomStateMax, randomStateMin=randomStateMin,
                                      solver=solver, targetMSE=targetRRMSE, n_delays = n_delays, n_pred =n_pred, normOut = normOut, stop=stop, learning_rate=learning_rate, alpha=alpha) 
            
            return render(request, 'MLP_results.html', {'idosorok': idosorok[request.session.session_key]})

    except Exception as e:
        print(traceback.format_exc())
        return HttpResponse("Hiba történt. "+str(e))
    
def AbrazolEgyben(adatsorok: list, idoszakok: list, megnevezesek: list, cim: str="", yFelirat:str="", grid:bool=False, num: int = 1):
    """Egy vagy több adatsor ábrázolása egy grafikonon.  Az x tengely feliratozásának sűrűsége dinamikusan állítódik be. """
    global idosorok

    try:
        # Határozzuk meg a legrövidebb idősor hosszát
        min_length = min(len(idoszakok), min(len(adatsor) for adatsor in adatsorok))

        # Vágjuk le az idősorokat a legrövidebb hosszra
        idoszakok = idoszakok[:min_length]
        adatsorok = [adatsor[:min_length] for adatsor in adatsorok]

        plt.figure(num=num, figsize=(20, 10))

        for i, idosor in enumerate(megnevezesek):
            plt.plot(idoszakok, adatsorok[i], label=idosor, linewidth=2.5)
        plt.ylabel(yFelirat)

        plt.title(cim)
        plt.grid(grid)

        # Lépésköz dinamikius meghatározása  alapján
        step = max(1, int(len(idoszakok) / 12))
        plt.xticks(range(0, len(adatsorok[0]), step), idoszakok[::step], rotation=45)  # x tengely feliratok beállítása
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        return buffer

    except Exception as e:
        print("Hiba a diagram készítése során!")
        print(traceback.format_exc())
       
def createObjects(idosorok_nevek, adatok, idoPontok, train_size):
    """A külön listákba kinyert oszlopokból Idősor példányok készítése a könnyebb kezelhetőség miatt.
     Az idősor felosztása tanító- és tesztadatokra a keresztvalidációhoz. """
    idosorok = []
    try:
        if(train_size < 50): train_size = 50
        if(train_size > 95): train_size = 95
        train_size_index = int(len(adatok[0]) * (train_size / 100)) 

        for i in range(len(idosorok_nevek)):     
            statisztika = Idosor(
                idosor_nev=idosorok_nevek[i],
                tanito_adatok=adatok[i][:train_size_index],
                teszt_adatok=adatok[i][train_size_index:],
                idoszakok=idoPontok[:train_size_index],
                teszt_idoszakok=idoPontok[train_size_index:]
            )
            idosorok.append(statisztika)
      
    except Exception as e:
        traceback.format_exc(e)
        print(f"beolvasott idősor hossza: {len(adatok[0])} \n elválasztó index: {train_size_index}")

    return idosorok

def arima(request):
    """Az ARIMA modellek paraméterezési oldalán levő űrlap beküldését követően 
    létrejönnek az idősor(ok)hoz rendelt regressziós modellek, azok be lesznek tanítva a paraméterek szerint
    és létrejönnek a tesztadatokra adott becslések + a kért előrejelzések.
    A tanulási és általánosítási folyamatot kiértékelő mutatók, táblázatok és grafikonokból létrejön a visszaküldendő weblap"""
    try: 
        global idosorok
        idosorok_ = []
        adatsorok =[] 
        n_pred = int(request.POST['n_pred'])

        for idosor in idosorok[request.session.session_key]:
            p = request.POST[idosor.idosor_nev+'_p']
            q = request.POST[idosor.idosor_nev+'_q']
            d = request.POST[idosor.idosor_nev+'_d']
            autoArima = False
            if (idosor.idosor_nev+'_autoArima' in request.POST):
                autoArima = True
            test_results = idosor.predictARIMA(p, d, q, n_pred, autoArima)

            if test_results:
                idosorok_.append(idosor.idosor_nev)
                adatsorok.append(idosor.ARIMA.becslesek)             
            else:
                print("hiba történt az ARIMA becslések készítése közben")
                
    
        adatsorok = []
        adatsorNevek = []

        for idosor in idosorok[request.session.session_key]:
            adatsorNevek.append(idosor.ARIMA.modelName)
            adatsorok.append(idosor.ARIMA.becslesek)

        try:
            idoszakok = idosor.teszt_idoszakok
            diagaramEgyben = AbrazolEgyben(adatsorok, idoszakok, adatsorNevek, idosorok[request.session.session_key][0].idosor_nev, "", grid=True, num=3)
            diagaramEgyben = base64.b64encode(diagaramEgyben.read()).decode('utf-8')
            return render(request, "ARIMA_results.html", {"idosorok": idosorok[request.session.session_key], "diagaramEgyben": diagaramEgyben})

        except Exception as e:
             print("Valami hiba történt")
             traceback.format_exc(e)
             return render(request, "ARIMA_results.html", {"idosorok": idosorok[request.session.session_key]})

    except:
        print("Valami hiba történt")
        print(traceback.format_exc())
        return redirect('home')