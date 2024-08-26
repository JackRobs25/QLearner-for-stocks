import argparse
from backtester import get_data
import pandas as pd
import numpy as np
from tech_ind import bollinger_bands
from tech_ind import relative_strength_index
from tech_ind import macd
from TabularQLearner import TabularQLearner
from DoubleQLearner import DoubleQLearner
from OracleStrategy import BaselineStrategy
from backtester import assess_strategy
from TechnicalStrategy import TechnicalStrategy
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler




class StockEnvironment:

  def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash
    self.learner = None

  # why did this have a data_folder parameter
  def prepare_world (self, start_date, end_date, symbol):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """

    data = get_data(start_date, end_date, [symbol], include_spy=False)
    world = pd.DataFrame(index=data.index, columns=['Price', 'MACD', 'RSI', 'Bollinger Band Percentage']) 
    bollinger_data = bollinger_bands(data[symbol], window=9, num_std=2)
    bollinger_data['percent'] = data[symbol] - bollinger_data['Lower Band'] / bollinger_data['Upper Band'] - bollinger_data['Lower Band']
    rsi_data = relative_strength_index(data)
    macd_data = macd(data, symbol)

    # Normalize bollinger_data['percent']
    bollinger_data_normalized = bollinger_data.copy()  # Make a copy to avoid modifying the original DataFrame
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    bollinger_data_normalized['percent'] = scaler.fit_transform(bollinger_data_normalized['percent'].values.reshape(-1, 1))

    # Normalize rsi_data
    rsi_data_normalized = rsi_data.copy()  # Make a copy to avoid modifying the original DataFrame
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    rsi_data_normalized[symbol] = scaler.fit_transform(rsi_data_normalized[symbol].values.reshape(-1, 1))

    # Normalize macd_data
    macd_data_normalized = macd_data.copy()  # Make a copy to avoid modifying the original DataFrame
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    macd_data_normalized['MACD'] = scaler.fit_transform(macd_data_normalized['MACD'].values.reshape(-1, 1))

    # print("bollinger: ", bollinger_data_normalized)
    # print("rsi: ", rsi_data_normalized)
    # print("macd: ", macd_data_normalized)


    world['Price'] = data[symbol]
    world['MACD'] = pd.qcut(macd_data_normalized['MACD'], 6, labels=[1,2,3,4,5,6])
    world['RSI'] = pd.qcut(rsi_data_normalized[symbol], 6, labels=[1,2,3,4,5,6])
    world['BBP'] = pd.qcut(bollinger_data_normalized['percent'], 6, labels=[1,2,3,4,5,6])
    return world


  
  def calc_state (self, df, day, holdings):
    """ Quantizes the state to a single number. """

    row = df.loc[day]
    hold = 0
    if holdings > 0: # long
      hold = 0
    elif holdings == 0: # flat
      hold = 1
    else: # short
      hold = 2
    BBP = row["Bollinger Band Percentage"] if not pd.isna(row["Bollinger Band Percentage"]) else 0
    RSI = row["RSI"] if not pd.isna(row["RSI"]) else 0
    MACD = row["MACD"] if not pd.isna(row["MACD"]) else 0
    return int(hold + BBP*6**2 + RSI*6 + MACD)
    

  
  def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     double = False, eps = 0.0, eps_decay = 0.0, print_results = True):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Only if print_results is True, print a summary result of what happened
    at the end of each trip.  Print nothing if print_results is False.

    If printing, feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """
    # select technical features (Bollinger bands, RSI, MACD)
    # compute values for data
    world = self.prepare_world(start, end, symbol)
    if double:
      learner = DoubleQLearner(states=157, actions=3, epsilon=eps, epsilon_decay=eps_decay, dyna=dyna)
    else:
      learner = TabularQLearner(states=157, actions=3, epsilon=eps, epsilon_decay=eps_decay, dyna=dyna)
    for i in range(trips):
      dates = pd.bdate_range(start, end)
      trades = pd.DataFrame(index=dates)
      trades['Trade'] = 0
      portfolio_value = pd.DataFrame(index=dates)
      portfolio_value['Value'] = 0
      cash = self.starting_cash
      holdings = 0
      prev_value = cash
      for index, row in world.iterrows(): # for each day in training data
        price = world.loc[index.strftime('%Y-%m-%d'), 'Price']
        curr_value = holdings*price + cash
        # compute current state
        curr_state = self.calc_state(world, index, holdings)
        # compute the reward for the previous action
        reward = curr_value - prev_value
        # print("reward: ", reward)
        # print("curr_state: ", curr_state)  
        # query learner to get an action
        action = learner.train(curr_state, reward)
        # long = 2
        # short = 1
        # flat = 0
        trade = 0
        share_limit = self.shares
        if action == 2: # buy
          if holdings > 0:
            trade = 0
          elif holdings == 0:
            trade = share_limit
          else:
            trade = share_limit*2
        elif action == 1: # sell
          if holdings > 0:
            trade = -share_limit*2
          elif holdings == 0:
            trade = -share_limit
          else:
            trade = 0
        elif action == 0: # flat
          if holdings > 0:
            trade = -share_limit
          elif holdings == 0:
            trade = 0
          else:
            trade = share_limit

        stock_value = abs(trade * price)
        if(not math.isnan(stock_value)):
          if trade > 0:
            fee = self.fixed_cost + (self.floating_cost * stock_value)
            cash -= fee
            cash -= stock_value
            holdings += trade
            
          if trade < 0:
            fee = self.fixed_cost + (self.floating_cost * stock_value)
            cash -= fee
            cash += stock_value
            holdings += trade

        trades.at[index, 'Trade'] = trade
        portfolio_value.at[index, 'Value'] = curr_value.astype('int64')
        prev_value = curr_value
        pd.set_option('display.max_columns', None)
        # print("port: ", portfolio_value)
      if print_results:
        print("Trip", i, "net result: ", portfolio_value.tail(1)['Value'].values[0])
    self.learner = learner

    


  
  def test_learner( self, start = None, end = None, symbol = None, print_results = True):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Only if print_results is True, print a summary result of what happened
    during the test.  Print nothing if print_results is False.

    If printing, feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000

    Return a tuple of the test trip net result and the benchmark result, in that order.
    """
    world = self.prepare_world(start, end, symbol)

    dates = pd.date_range(start=start, end=end)
    trade_df = pd.DataFrame(index=dates, columns=['Trade'])
    spy_df = pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'])
    spy_df.rename(columns={'Adj Close': 'SPY'}, inplace=True)
    learned_trades = trade_df.join(spy_df, how='inner')
    learned_trades.drop('SPY', axis=1, inplace=True)
    learned_trades['Trade'] = 0

    portfolio_value = pd.DataFrame(index=dates)
    portfolio_value['Value'] = 0

    cash = self.starting_cash
    holdings = 0


    for index, row in world.iterrows():
      price = world.loc[index.strftime('%Y-%m-%d'), 'Price']
      curr_value = holdings*price + cash
      curr_state = self.calc_state(world, index, holdings)
      action = self.learner.test(curr_state)

      trade = 0
      share_limit = self.shares
      if action == 2: # buy
        if holdings > 0:
          trade = 0
        elif holdings == 0:
          trade = share_limit
        else:
          trade = share_limit*2
      elif action == 1: # sell
        if holdings > 0:
          trade = -share_limit*2
        elif holdings == 0:
          trade = -share_limit
        else:
          trade = 0
      elif action == 0: # flat
        if holdings > 0:
          trade = -share_limit
        elif holdings == 0:
          trade = 0
        else:
          trade = share_limit

      stock_value = abs(trade * price)
      if(not math.isnan(stock_value)):
        if trade > 0:
          fee = self.fixed_cost + (self.floating_cost * stock_value)
          cash -= fee
          cash -= stock_value
          holdings += trade
          
        if trade < 0:
          fee = self.fixed_cost + (self.floating_cost * stock_value)
          cash -= fee
          cash += stock_value
          holdings += trade
              
      learned_trades.at[index, 'Trade'] = trade
      portfolio_value.at[index, 'Value'] = curr_value.astype('int64')
      
    # Compute benchmark result
    baseline = BaselineStrategy()
    baseline_strategy = baseline.test(start_date=start, end_date=end)
    b_ADR, b_CR, b_SD, b_DCR = assess_strategy(symbol, baseline_strategy, starting_value=self.starting_cash, fixed_cost=self.fixed_cost, floating_cost=self.floating_cost)
    technical = TechnicalStrategy()
    technical_trades, long, short = technical.test(start_date=start, end_date=end)

    print("port: ", portfolio_value)

    test_trip_net_result = portfolio_value.tail(1)
    benchmark_result = (b_CR+1)*self.starting_cash

    if print_results:
      print("Test trip, net result: ", test_trip_net_result['Value'].values[0])
      print("Benchmark result: ",  benchmark_result)
    
    return test_trip_net_result, benchmark_result, technical_trades, learned_trades
  

if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.

  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--double', action='store_true', help='If supplied, use Double Q.')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--test_oos', action='store_true', help='If supplied, run out of sample tests.')
  sim_args.add_argument('--trials', default=1, type=int, help='Number of complete experimental trials.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data per trial.')

  args = parser.parse_args()

  # Store the final in-sample and out-of-sample result of each trial.
  is_results = []
  oos_results = []


  # Run potentially many experiments.
  for trial in range(args.trials):

    # Create an instance of the environment class.
    env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                            share_limit = args.shares )

    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                       symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                       double = args.double, eps = args.eps, eps_decay = args.eps_decay )

    # Test the learned policy and see how it does.

    # In sample.
    is_result, is_bench, tech_trades, learned_trades = env.test_learner( start = args.train_start, end = args.train_end, symbol = args.symbol )
    is_results.append(is_result)

    # print("tech_trades: ", tech_trades)
    # print("tech_trades.index: ", tech_trades.index)

    baseline = BaselineStrategy()
    baseline_strategy = baseline.test(symbol=args.symbol, start_date=args.train_start, end_date=args.train_end, starting_cash=args.cash)
    b_ADR, b_CR, b_SD, b_DCR = assess_strategy(args.symbol, baseline_strategy, starting_value=args.cash, fixed_cost=args.fixed, floating_cost=args.floating)
    t_ADR, t_CR, t_SD, t_DCR = assess_strategy(args.symbol, tech_trades, starting_value=args.cash, fixed_cost=args.fixed, floating_cost=args.floating)
    l_ADR, l_CR, l_SD, l_DCR = assess_strategy(args.symbol, learned_trades, starting_value=args.cash, fixed_cost=args.fixed, floating_cost=args.floating)
    # Plotting Tech Strategy vs Baseline Strategy
    plt.figure(figsize=(10, 6))
    plt.plot(t_DCR, label='Tech Strategy', color='blue')
    plt.plot(b_DCR, label='Baseline Strategy', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Tech Strategy vs Baseline Strategy - Cumulative Returns')
    plt.legend()
    plt.show()

    print("cumulative technical return: ", t_DCR.tail(1))

    # Plotting Learned Strategy vs Baseline Strategy
    plt.figure(figsize=(10, 6))
    plt.plot(l_DCR, label='Learned Strategy', color='green')
    plt.plot(b_DCR, label='Baseline Strategy', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Learned Strategy vs Baseline Strategy - Cumulative Returns')
    plt.legend()
    plt.show()

    print("cumulative learned return: ", l_DCR.tail(1))




    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    if args.test_oos:
      oos_result, oos_bench = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )
      oos_results.append(oos_result)


  # Print summary results.
  is_results = np.array(is_results)
  if args.test_oos: oos_results = np.array(oos_results)

  is_min, is_med, is_max = np.min(is_results), np.median(is_results), np.max(is_results)
  if args.test_oos: oos_min, oos_med, oos_max = np.min(oos_results), np.median(oos_results), np.max(oos_results)

  print (f"\nAfter {args.trials} trials of {args.trips} trips:\n");
  print (f"In-sample net result min {is_min:.2f}, median {is_med:.2f}, max {is_max:.2f}, vs benchmark {is_bench:.2f}")

  if args.test_oos:
    print (f"Out-of-sample net result min {oos_min:.2f}, median {oos_med:.2f}, max {oos_max:.2f}, vs benchmark {oos_bench:.2f}")

