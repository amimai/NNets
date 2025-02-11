### fxcmpy_handler ###
## connectd to fxcmpy demo account and fetches data ##
import datetime as dt
import time

# from matplotlib import pyplot  as plt
# import pandas as pd
# import numpy as np

# your token here #
TOKEN = '85902b8e33d2198b8a99769de6e820b85a42e4b7' #None
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

def get_months(timespan, instrument, period, con=None):
    delta = 30
    if con is None:
        connection = get_con()
    else: connection = con
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

def get_more_data(instruments, timespan,period,con=None):
    data = []
    for instrument in instruments:
        print ('starting {0}'.format(instrument))
        while True:
            try:
                temp = get_months(timespan, instrument=instrument, period = period, con=con)
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


if __name__ == '__main__':
    con = get_con()
    instruments = con.get_instruments()
    period = 'm5'
    delta = 30
    data = []
    now = con.get_candles(instrument=instruments[0], number=1).index[0]
    end = now - dt.timedelta(delta)
    for i in range(len(instruments)):
        data.append(con.get_candles(instrument=instruments[i], period=period,
                                    columns=['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty'],
                                    start=end, end=now))
    edat = []
    edat.append([each.shape for each in data])
    print(edat[-1])
    print(data[0].index[0])
    print(data[0].index[-1])
    now = data[0].index[0]
    data2 = []
    end = now - dt.timedelta(delta)
    for i in range(len(instruments)):
        data2.append(con.get_candles(instrument=instruments[i], period=period,
                                    columns=['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty'],
                                    start=end, end=now))

    edat.append([each.shape for each in data2])
    print(edat[-1])
    print(data2[0].index[0])
    print(data2[0].index[-1])

    dat_out = data
    dat_out = [data2[i].append(dat_out[i]).drop_duplicates() for i in range(len(dat_out))]
    print(dat_out[0].index[0])
    print(dat_out[0].index[-1])

    for _ in range(int(4*360/delta)):
        data3 = []
        now = dat_out[0].index[0]
        end = now - dt.timedelta(delta)
        for i in range(len(instruments)):
            data3.append(con.get_candles(instrument=instruments[i], period=period,
                                         columns=['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty'],
                                         start=end, end=now))
        dat_out = [data3[i].append(dat_out[i]).drop_duplicates() for i in range(len(dat_out))]
        edat.append([each.shape for each in data3])
        print(data3[0].index[0])
        print(data3[0].index[-1])


    ret = dat_out[0].join(dat_out[1], lsuffix=instruments[0], rsuffix=instruments[1])
    for i in range(2,len(dat_out)):
        ret = ret.join(dat_out[i],lsuffix='',rsuffix=instruments[i])
    ret.rename(columns={'bidopen':'bidopen'+instruments[2],
                        'bidclose':'bidclose'+instruments[2],
                        'bidhigh':'bidhigh'+instruments[2],
                        'bidlow': 'bidlow'+instruments[2],
                        'tickqty':'tickqty'+instruments[2]}, errors="raise")


    file_data = 'all_data_223k_3y_m5.csv'
    import pandas as pd
    dat = pd.read_csv(file_data, index_col='date')
    print(dat[0].index[0])
    print(dat[0].index[-1])

    dat_exp = dat.append(ret).drop_duplicates()
    print(dat_exp[0].index[0])
    print(dat_exp[0].index[-1])

    dat_exp.to_csv('all_data_234k_4y_m5.csv')




    for i in range(len(instruments)):
        print(instruments[i])
    try:
        con.get_candles(instrument=str(instruments[i]), number=1)
    except Exception as e:
        print(e)


