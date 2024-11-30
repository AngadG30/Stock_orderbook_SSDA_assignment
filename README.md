# Stock_orderbook_SSDA_assignment
Group assignment for Simulation- Angad, Paras

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

class MarkovOrderbookComparator:
    def __init__(self, symbol, states=5):
        self.symbol = symbol
        self.states = states
        self.transition_matrix = np.zeros((states, states))

    def fetch_historical_data(self, period='6mo'):
        stock_data = yf.download(self.symbol, period=period)
        return stock_data

    def discretize_prices(self, prices):
        price_min, price_max = prices.min(), prices.max()
        return np.floor(((prices - price_min) / (price_max - price_min)) * (self.states - 1)).astype(int)

    def compute_transition_probabilities(self, prices):
        for i in range(len(prices) - 1):
            self.transition_matrix[prices[i], prices[i+1]] += 1
        self.transition_matrix += 1
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

    def validate_markov_model(self, test_prices):
        test_states = self.discretize_prices(test_prices)
        observed_transitions = np.zeros_like(self.transition_matrix)
        for i in range(len(test_states) - 1):
            observed_transitions[test_states[i], test_states[i+1]] += 1
        observed_transitions += 1e-10

        try:
            chi2_stat, p_value = stats.chi2_contingency(observed_transitions)[:2]
            return {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'model_acceptance': p_value > 0.05
            }
        except ValueError:
            return {
                'chi2_statistic': None,
                'p_value': None,
                'model_acceptance': False,
                'error': 'Insufficient state transitions for statistical validation'
            }

    def compute_ask_bid_spread(self, stock_data):
        ask_prices = stock_data['High']
        bid_prices = stock_data['Low']
        spread = ask_prices - bid_prices
        return ask_prices, bid_prices, spread

    def plot_ask_bid_histogram(self, ask_prices, bid_prices):
        bin_min = float(bid_prices.min())  # Extract scalar value
        bin_max = float(ask_prices.max())  # Extract scalar value
        bins = np.linspace(bin_min, bin_max, 30)

        plt.figure(figsize=(12, 6))
        plt.hist(bid_prices, bins=bins, alpha=0.7, label='Bid Prices (Buy Orders)', color='blue', edgecolor='black')
        plt.hist(ask_prices, bins=bins, alpha=0.7, label='Ask Prices (Sell Orders)', color='green', edgecolor='black')
        plt.title(f'Ask and Bid Price Distribution for {self.symbol}')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def compare_real_time_data(self):
        stock_data = self.fetch_historical_data()
        prices = stock_data['Close'].values

        train_prices = prices[:len(prices)//2]
        test_prices = prices[len(prices)//2:]

        discretized_train = self.discretize_prices(train_prices)
        self.compute_transition_probabilities(discretized_train)

        validation_results = self.validate_markov_model(test_prices)
        ask_prices, bid_prices, spread = self.compute_ask_bid_spread(stock_data)

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, spread, label='Ask-Bid Spread', color='orange')
        plt.title(f'Ask-Bid Spread for {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.plot_ask_bid_histogram(ask_prices, bid_prices)

        return {
            'historical_prices': stock_data['Close'],
            'transition_matrix': self.transition_matrix,
            'validation': validation_results,
            'ask_prices': ask_prices,
            'bid_prices': bid_prices,
            'ask_bid_spread': spread
        }

def main():
    try:
        comparator = MarkovOrderbookComparator('AAPL', states=5)
        results = comparator.compare_real_time_data()

        print("Transition Matrix:")
        print(results['transition_matrix'])
        print("\nValidation Results:")
        for k, v in results['validation'].items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
