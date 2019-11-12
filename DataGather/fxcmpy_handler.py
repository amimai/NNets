### fxcmpy_handler ###
## connectd to fxcmpy demo account and fetches data ##

# your token here #
TOKEN = 0
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

def get_market(con, quantity,clean=True):
    instruments = con.get_instruments()
    data_init = ticks(con, instrument=instruments[0], number=quantity)
    # hkg33, cryptomajor
    data_init = data_init.rename(columns={'bidopen': 'bidopen' + instruments[0],
                      'bidclose': 'bidopen' + instruments[0],
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
        data_init = data_init.fillna(method='bfill')
        data_init = data_init.fillna(method='ffill')

    return data_init

