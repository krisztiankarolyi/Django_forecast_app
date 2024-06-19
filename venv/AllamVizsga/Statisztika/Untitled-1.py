def rollingForecast(self, data):
    """Rekurzív becslés a teszt adatokra: a modell minden egyes becslés után újrailleszti önmagát.
    Sokkal pontosabb, mint egy lépésben becsülni több értéket, azonban erőforrás igényes lehet."""
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        
    return predictions