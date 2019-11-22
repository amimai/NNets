### fxcmpy_handler ###
## connectd to fxcmpy demo account and fetches data ##
import private
import datetime as dt
import time

# from matplotlib import pyplot  as plt
# import pandas as pd
# import numpy as np

# your token here #
TOKEN = private.token
# your token here #

colums = ['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty']

import fxcmpy

# connect #
def get_con():
    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
    return con

# generic get function #
def ticks(con,instrument='USD/JPY', columns=['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty'], period='H1', number=10):
    data = con.get_candles(instrument, columns=columns, period=period, number=number)
    return data

# for recent data with all open markets, tail is full of bad and uneven data due to API
def get_market(con, quantity,clean=True):
    instruments = con.get_instruments()
    data_init = ticks(con, instrument=instruments[0], number=quantity)
    # hkg33, cryptomajor
    data_init = data_init.rename(columns={'bidopen': 'bidopen' + instruments[0],
                      'bidclose': 'bidclose' + instruments[0],
                      'bidhigh': 'bidhigh' + instruments[0],
                      'bidlow': 'bidlow' + instruments[0],
                      'tickqty': 'tickqty' + instruments[0]})
    instruments.pop(0)

    # iterate and join #
    for instrument in instruments:
        data = ticks(con, instrument=instrument, number=quantity)
        data_init = data_init.join(data,rsuffix=instrument)

    # clean up data #
    if clean:
        data_init = data_init.fillna(method='ffill')
        data_init = data_init.fillna(method='bfill')

    return data_init


def get_timeperiod(connection,start,end, instrument, period):
    data = connection.get_candles(instrument=instrument, period = period,
                                  columns = ['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty'],
                                  start=start, end=end)
    return data

def get_months(timespan, instrument, period):
    delta = 30
    connection = get_con()
    now = connection.get_candles(instrument=instrument,number=1).index[0]  ## get a timestamp

    data = []

    for i in range(timespan):
        dataset = get_timeperiod(connection, now - dt.timedelta(delta),
                       now, instrument=instrument,period=period)
        time.sleep(.5)
        now = now - dt.timedelta(delta)

        data.append(dataset)

    ret = data[0]
    for each in data[1:]:
        ret = each.append(ret).drop_duplicates()
    connection.close()
    return ret

def get_more_data(instruments, timespan,period):
    data = []
    for instrument in instruments:
        print ('starting {0}'.format(instrument))
        while True:
            try:
                temp = get_months(timespan, instrument=instrument, period = period)
                print (temp.shape)
                #print (temp.)
                data.append(temp)
                temp = 0
                break
            except Exception as e:
                print (e)
                time.sleep(10)
                try:
                    print('restarted')
                    continue
                except:
                    continue
    ret = data[0].join(data[1],lsuffix=instruments[0],rsuffix=instruments[1])
    for i in range(2,len(data)):
        ret = ret.join(data[i],rsuffix=instruments[i])
    return ret, data
