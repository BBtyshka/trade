from statsmodels.tsa.arima.model import ARIMA
from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
import tensorflow as tf
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from datetime import datetime, timedelta
import talib, keras
import numpy as np

class AlgoEvent:
    def __init__(self):
        self.lasttradetime = datetime(2000,1,1)
        self.balance = 0
        self.count = 0
        self.traintime = datetime(2021,1,1)
        self.history = []
        self.params = (4,1,2)

    def start(self, mEvt):
        self.myinstrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()
        self.model = self.evt.path_lib+'ARIMA_AAPL_1'

    def on_bulkdatafeed(self, isSync, bd, ab):
        self.balance = ab['availableBalance']
        if bd[self.myinstrument]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
            self.lasttradetime = bd[self.myinstrument]['timestamp']
            lastprice = bd[self.myinstrument]['lastPrice']
            self.history.append(lastprice)
            
            if bd[self.myinstrument]['timestamp'] > self.traintime:
                model = ARIMA(self.history, order=self.params)
                model_fit = model.fit()
                prediction = model_fit.forecast(step=1)[0]
            
                if prediction > lastprice:
                    #buy
                    dif = prediction-lastprice
                    self.evt.consoleLog(f"Buy at {lastprice}, dif: {dif}")
                    self.test_sendOrder(lastprice, 1, 'open', dif)
                else:
                    dif = lastprice-prediction
                    self.evt.consoleLog(f"Sell at {lastprice}, dif: {dif}")
                    self.test_sendOrder(lastprice, -1, 'open', dif)
    
    def test_sendOrder(self, lastprice, buysell, openclose, dif):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.myinstrument
        order.orderRef = 1
        dif = abs(dif)
        
        
        if buysell==1:
            order.takeProfitLevel = lastprice*1.2+dif
            order.stopLossLevel = lastprice*0.9
        elif buysell==-1:
            order.takeProfitLevel = lastprice*0.95-dif
            order.stopLossLevel = lastprice*1.1
        
        order.volume = dif*50
        order.openclose = openclose
        order.buysell = buysell
        order.ordertype = 0 #0=market_order, 1=limit_order, 2=stop_order
        self.evt.sendOrder(order)