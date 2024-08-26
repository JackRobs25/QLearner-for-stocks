import pandas as pd
import math
import numpy as np
############### backtester code ##################
def assess_strategy(symbol, trades, starting_value = 200000, fixed_cost = 9.95, floating_cost = 0.005):
    # symbol = 'DIS' # can only trade disney 
    
    dates = trades.index.tolist()
    portfolio_value = []
    df = get_data(dates[0], dates[-1], [symbol], column_name="Adj Close", data_folder="./data", include_spy=False)
    
    cash = starting_value
    current_shares = 0
    for index, row in trades.iterrows():
        shares = row['Trade']
        price = df.loc[index.strftime('%Y-%m-%d')].values
        stock_value = abs(shares * price)
        if(not math.isnan(stock_value)):
            if shares > 0:
                fee = fixed_cost + (floating_cost * stock_value)
                cash -= fee
                cash -= stock_value
                current_shares += shares
                
            if shares < 0:
                fee = fixed_cost + (floating_cost * stock_value)
                cash -= fee
                cash += stock_value
                current_shares += shares


        portfolio_value.append(cash + (current_shares*price))

    trade_df = pd.DataFrame(portfolio_value, index=dates)
    trade_df.columns = ["Portfolio Price"]  


    daily_portfolio_values = trade_df

    SR, ADR, CR, SD, final, DCR = calculate_info(daily_portfolio_values)

    return ADR, CR, SD, DCR



def get_adj_close(date, symbol, column_name="Adj Close", data_folder="./data"):
    df = pd.read_csv('Data/' + symbol + '.csv')
    row = df[df['Date'].str.strip() == date.date().strftime('%Y-%m-%d')]
    if row.empty:
        return 0 
    adj_close = row[column_name].values[0]
    return adj_close

def calculate_info(daily_prices_df, risk_free_rate=0, sample_freq=252):
    # Multiply each column by the allocation to that stock
    cumulative_return = (daily_prices_df.iloc[-1] / daily_prices_df.iloc[0]) - 1
    # pct_change() --> Fractional change between the current and a prior element
    daily_returns = daily_prices_df.pct_change().dropna()
    daily_cumulative_returns = (1 + daily_returns).cumprod() - 1
    average_daily_return = daily_returns.mean()
    stdev_daily_return = daily_returns.std()
    # Calculate Sharpe Ratio
    excess_daily_returns = daily_returns - risk_free_rate
    sharpe_ratio = (excess_daily_returns.mean() / excess_daily_returns.std()) * math.sqrt(sample_freq)
    end_value = daily_prices_df.iloc[-1]
    return sharpe_ratio.values[0], average_daily_return.values[0], cumulative_return.values[0], stdev_daily_return.values[0], end_value.values[0], daily_cumulative_returns

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

def assess_portfolio (start_date, end_date, symbols, allocations,
                      starting_value=1000000, risk_free_rate=0.0,
                      sample_freq=252, plot_returns=True):
    daily_prices_df = get_data(start_date, end_date, symbols)
    # Normalize stock prices to the first day
    normalized_prices = daily_prices_df / daily_prices_df.iloc[0]
    # Multiply each column by the allocation to that stock
    allocated_prices = normalized_prices.iloc[:, 1:] * allocations
    # Multiply normalized allocations by starting portfolio dollar value
    portfolio_values = allocated_prices * starting_value
    # Sum each date (across the stocks) to get daily portfolio dollar value
    daily_portfolio_value = portfolio_values.sum(axis=1)
    cumulative_return = (daily_portfolio_value.iloc[-1] / daily_portfolio_value.iloc[0]) - 1
    # pct_change() --> Fractional change between the current and a prior element
    daily_returns = daily_portfolio_value.pct_change().dropna()
    average_daily_return = daily_returns.mean()
    stdev_daily_return = daily_returns.std()
    # Calculate Sharpe Ratio
    excess_daily_returns = daily_returns - risk_free_rate
    sharpe_ratio = (excess_daily_returns.mean() / excess_daily_returns.std()) * math.sqrt(sample_freq)
    end_value = daily_portfolio_value.iloc[-1]


    return sharpe_ratio, average_daily_return, cumulative_return, stdev_daily_return, end_value 

###########################################################
