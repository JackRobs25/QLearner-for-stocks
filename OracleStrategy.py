import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from backtester import assess_strategy

def get_data(start, end, symbols, column_name="Adj Close", include_spy=True, data_folder="./data"):
    dates = pd.date_range(start, end)
    df1 = pd.DataFrame(index=dates)
    df2 = pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', column_name])
    df2.rename(columns={column_name: "SPY"}, inplace=True)
    df1 = df1.join(df2, how='inner')
    for symbol in symbols:
        tmp_df = pd.read_csv(data_folder + '/'+ symbol + ".csv", index_col='Date', parse_dates=True, usecols=['Date', column_name])
        tmp_df.rename(columns={column_name: symbol}, inplace=True)
        df1 = df1.join(tmp_df, how='left', rsuffix='_'+symbol)
    if (not include_spy):
        df1.drop('SPY', axis=1, inplace=True)
    return df1

class OracleStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(self, start_date = '2018-01-01', end_date = '2019-12-31', symbol = 'DIS', starting_cash = 200000):
        # Inputs represent the date range to consider, the single stock to trade, and the starting portfolio value.
        #
        # Return a date-indexed DataFrame with a single column containing the desired trade for that date.
        # Given the position limits, the only possible values are -2000, -1000, 0, 1000, 2000.
        
        dates = pd.date_range(start=start_date, end=end_date) 
        df_trades = pd.DataFrame(index=dates)
        df_stock = get_data(start_date, end_date, [symbol], include_spy=False)
        df_trades = df_trades.join(df_stock, how='inner')
        df_trades = df_trades.rename(columns={symbol:"Trade"})
        df_trades = df_trades.diff().shift(-1) # this subtracts tomorrows price from today so show what will happen the next day

        shares_held = 0

        for index, row in df_trades.iterrows():
            diff = row['Trade']
            if diff > 0: #BUY
                if shares_held > 0:
                    diff = 0 
                if shares_held < 0: 
                    diff = 2000
                    shares_held += diff
                if shares_held == 0: 
                    diff = 1000
                    shares_held += diff

            elif diff < 0: #SELL
                if shares_held < 0:
                    diff = 0 
                if shares_held > 0:
                    diff = -2000
                    shares_held += diff
                if shares_held == 0:
                    diff = -1000
                    shares_held += diff

            df_trades.at[index, 'Trade'] = diff

        return df_trades


    
class BaselineStrategy:
    def __init__(self, *params, **kwparams):
        pass

    def train(self, *params, **kwparams):
        pass

    def test(self, start_date = '2018-01-01', end_date = '2019-12-31', symbol = 'DIS', starting_cash = 200000):
        dates = pd.date_range(start=start_date, end=end_date)
        trade_df = pd.DataFrame(index=dates, columns=['Trade'])
        spy_df = pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'])
        spy_df.rename(columns={'Adj Close': 'SPY'}, inplace=True)
        df_trades = trade_df.join(spy_df, how='inner')
        df_trades.drop('SPY', axis=1, inplace=True)

        df_trades['Trade'] = 0
        df_trades['Trade'].iloc[0] = 1000
        df_trades.index.name = 'Date'


        return df_trades
    

def main():
    oracle = OracleStrategy()
    oracle_strategy = oracle.test()
    baseline = BaselineStrategy()
    baseline_strategy = baseline.test()
    o_ADR, o_CR, o_SD, o_DCR = assess_strategy(oracle_strategy, starting_value=200000, fixed_cost=0, floating_cost=0)
    b_ADR, b_CR, b_SD, b_DCR = assess_strategy(baseline_strategy, starting_value=200000, fixed_cost=0, floating_cost=0)
    plt.figure(figsize=(10, 6))
    plt.plot(o_DCR, label='Oracle Strategy', color='blue')
    plt.plot(b_DCR, label='Baseline Strategy', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Oracle Strategy vs Baseline Strategy - Cumulative Returns')
    plt.legend()
    plt.show()
    print("Cumulative return for the baseline strategy: ", b_CR)
    print("Cumulative return for the oracle strategy: ", o_CR)
    print("Average Daily Returns return for the baseline strategy: ", b_ADR)
    print("Average Daily Returns for the oracle strategy: ", o_ADR)
    print("Standard Deviation of daily returns for the baseline strategy: ", b_SD)
    print("Standard Deviation of daily returns for the oracle strategy: ", o_SD)





if __name__ == "__main__":
    main()


