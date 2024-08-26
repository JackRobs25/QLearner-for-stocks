import pandas as pd
import numpy as np
from OracleStrategy import BaselineStrategy
import math
import matplotlib.pyplot as plt
from tech_ind import simple_moving_average
from tech_ind import bollinger_bands
from tech_ind import relative_strength_index
from backtester import assess_strategy
from tech_ind import macd



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


class TechnicalStrategy:
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

        data = get_data(start_date, end_date, [symbol], include_spy=False)
        # df_trades = pd.DataFrame(index=data.index, columns=['Trade'])
        # df_decisions = pd.DataFrame(index=data.index, columns=['MACD', 'RSI', 'Bollinger']) # fill with short or long strings

        dates = pd.date_range(start=start_date, end=end_date)
        trade_df = pd.DataFrame(index=dates, columns=['Trade'])
        decision_df = pd.DataFrame(index=dates, columns=['MACD', 'RSI', 'Bollinger'])
        spy_df = pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'])
        spy_df.rename(columns={'Adj Close': 'SPY'}, inplace=True)
        df_trades = trade_df.join(spy_df, how='inner')
        df_decisions = decision_df.join(spy_df, how='inner')
        df_trades.drop('SPY', axis=1, inplace=True)
        df_decisions.drop('SPY', axis=1, inplace=True)


        # LONG = 1
        # GO_FLAT = 0
        # SHORT = -1

        bollinger_data = bollinger_bands(data[symbol], window=9, num_std=3)
        rsi_data = relative_strength_index(data)
        macd_data = macd(data, symbol)

        
        prev_macd = None
        signal_line = 2.3
        for date, row in macd_data.iterrows():
            curr_macd = macd_data.loc[date, 'MACD']
            if prev_macd == None:
                prev_macd = curr_macd
                continue
            # when the macd cross the zero line from negative to positive, SELL
            if prev_macd < signal_line and curr_macd > signal_line: 
                df_decisions.loc[date,'MACD'] = 1
            elif prev_macd > signal_line and curr_macd < signal_line: # SELL
                df_decisions.loc[date,'MACD'] = -1
            else:
                df_decisions.loc[date,'MACD'] = 0
            prev_macd = curr_macd
        

        for date, row in rsi_data.iterrows():
            if row['DIS'] > 87:
                df_decisions.loc[date,'RSI'] = -1
            elif row['DIS'] < 32:
                df_decisions.loc[date,'RSI'] = 1
            else:
                df_decisions.loc[date,'RSI'] = 0



        # go through each indicator separately and make an array for what they would do i.e long, short, nothing
        # then can loop through all of these and figure out final decision off of aggregate of what the indicators suggest
        long_positions = []
        short_positions = []
        shares_held = 0
        for date,row in df_decisions.iterrows():
            verdict = row['MACD'] + row['RSI'] 
            if verdict > 0: # LONG
                if shares_held > 0:
                    df_trades.loc[date] = 0
                elif shares_held == 0:
                    df_trades.loc[date] = 1000
                    long_positions.append(date)
                elif shares_held < 0:
                    df_trades.loc[date] = 2000
                    long_positions.append(date)
                shares_held = 1000
            elif verdict < 0: # SHORT
                if shares_held > 0:
                    df_trades.loc[date] = -2000
                    short_positions.append(date)
                elif shares_held == 0:
                    df_trades.loc[date] = -1000
                    short_positions.append(date)
                elif shares_held < 0:
                    df_trades.loc[date] = 0
                shares_held = -1000
            else:
                df_trades.loc[date] = 0

        return df_trades, long_positions, short_positions
    
def main():
    tech = TechnicalStrategy()
    # tech_strategy, long_pos, short_pos = tech.test()
    oos_tech_strat, long_pos, short_pos = tech.test(start_date='2020-01-01', end_date='2021-12-31')
    o_ADR, o_CR, o_SD, o_DCR = assess_strategy(oos_tech_strat, starting_value=200000)
    baseline = BaselineStrategy()
    # baseline_strategy = baseline.test()
    baseline_strategy = baseline.test(start_date='2020-01-01', end_date='2021-12-31')
    b_ADR, b_CR, b_SD, b_DCR = assess_strategy(baseline_strategy, starting_value=200000)
    plt.figure(figsize=(10, 6))
    plt.plot(o_DCR, label='My Strategy', color='blue')
    plt.plot(b_DCR, label='Baseline Strategy', color='orange')
    for date in long_pos:
        plt.axvline(x=date, color='green', linestyle='--', linewidth=1)
    for date in short_pos:
        plt.axvline(x=date, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('My Strategy vs Baseline Strategy - Cumulative Returns')
    plt.legend()
    plt.show()
    print("Cumulative return for the baseline strategy: ", b_CR)
    print("Cumulative return for the my strategy: ", o_CR)
    print("Average Daily Returns return for the baseline strategy: ", b_ADR)
    print("Average Daily Returns for the my strategy: ", o_ADR)
    print("Standard Deviation of daily returns for the baseline strategy: ", b_SD)
    print("Standard Deviation of daily returns for the my strategy: ", o_SD)





if __name__ == "__main__":
    main()
