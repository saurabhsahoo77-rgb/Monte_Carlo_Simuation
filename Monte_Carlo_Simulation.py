import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
import numpy.random as rdm

# Monte Carlo simulation class for portfolio VaR and CVaR estimation
class Monte_Carlo:
  def __init__(self, ticker, start_date, end_date, weights, alpha, sim, horizon_days, seed):
    self.ticker = ticker
    self.start_date = start_date
    self.end_date = end_date
    self.ticker = ticker
    self.weights = weights
    self.alpha = alpha                # Confidence level (e.g. 0.05 for 95% VaR)
    self.sim = sim                    # Number of Monte Carlo simulations
    self.horizon_days = horizon_days  # Risk horizon (e.g. 1-day VaR)
    self.seed = seed                  # Random seed for reproducibility

  # Download historical price data from Yahoo Finance
  def download_data(self):
    data = yf.download(self.ticker, start=self.start_date, end=self.end_date).dropna()

    if isinstance(self.ticker, str):
        # Handle single ticker case
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']]
        else:
            data = data[['Close']]
    else:
        # Handle multiple tickers
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        else:
            data = data['Close']
    return data

  # Calculate daily log returns
  def return_calculator(self):
    data = self.download_data()
    returns = np.log(data / data.shift(1)).dropna()
    return returns

  # Plot return distributions and compare with normal fit
  def exploratory_analysis(self):
    returns = self.return_calculator()
    data = self.download_data()
    mean = returns.mean()
    std = returns.std()

    for col in returns.columns:
      plt.figure(figsize=(10, 4))
      plt.hist(returns[col], bins=80, density=True, alpha=0.6, color='b')
      x = np.linspace(returns[col].min(), returns[col].max(), 200)
      plt.plot(x, stats.norm.pdf(x, mean[col], std[col]), color='r')
      plt.title(f'{col} — Empirical Returns vs Normal Fit')
      plt.xlabel('Daily log-return')
      legend = ['Normal', 'Empirical']
      plt.legend(legend, loc='upper left')
      print(f'{col} mean: {mean[col]*100:.2f}%')
      print(f'{col} std: {std[col]*100:.2f}%')
      plt.show()

  # Historical simulation approach for VaR estimation
  def Historical_VaR(self):
    weights = np.array(self.weights)
    weights = np.array(weights / weights.sum())  # Normalize weights
    returns = self.return_calculator()
    portfolio = returns.dot(weights)

    # Compute percentile at alpha level
    var = np.percentile(portfolio, self.alpha * 100)

    plt.hist(portfolio, bins=80)
    plt.axvline(var, color='r', linestyle='dashed', linewidth=1)
    legend = ['Historical Distribution', 'VaR']
    plt.legend(legend, loc='upper left')
    plt.title('Historical VaR')
    plt.show()

    return var * -100  # Convert to positive loss percentage

  # Monte Carlo simulation of portfolio returns
  def Monte_Carlo(self):
    rdm.seed(self.seed)
    returns = self.return_calculator()
    mu_vec = returns.mean()
    cov_mat = returns.cov()

    # Scale mean and covariance for horizon
    mu_h = mu_vec * self.horizon_days
    cov_h = cov_mat * self.horizon_days

    # Simulate correlated random returns
    simulated = rdm.multivariate_normal(mu_h, cov_h, self.sim)
    sim_port = simulated.dot(self.weights)
    return sim_port

  # Calculate VaR and CVaR using simulated returns
  def VaR_MC(self):
    loss = -self.Monte_Carlo()
    VaR = np.percentile(loss, self.alpha * 100)
    cVaR = loss[loss <= VaR].mean()
    return VaR, cVaR

  # Compare simulated vs historical return distributions
  def Visual(self):
    sim_returns = self.Monte_Carlo()
    plt.figure(figsize=(10, 4))
    plt.hist(sim_returns, bins=80, density=True, color='r', alpha=0.6)
    plt.hist(self.return_calculator().dot(self.weights), bins=80, color='b', density=True, alpha=0.7)
    plt.axvline(self.VaR_MC()[0], color='r', linestyle='dashed')
    plt.axvline(self.VaR_MC()[1], color='g', linestyle='dashed')
    legend = (((1 - self.alpha), 'VaR'), ((1 - self.alpha), 'cVaR'), 'Simulated Return', 'Historical Returns')
    plt.legend(legend, loc='upper right')
    plt.title('Simulated vs Historical Returns Distribution')
    plt.show()

  # Simple backtesting — count how often historical losses exceed VaR
  def back_test(self):
    portfolio_return = self.return_calculator().dot(self.weights)
    days = len(portfolio_return)
    VaR = self.VaR_MC()[0]
    loss_exceeded_VaR = portfolio_return[portfolio_return < VaR]
    print(f'\nHistorical days: {days}')
    print(f'No. of days loss exceeded VaR: {len(loss_exceeded_VaR)}')
    return loss_exceeded_VaR


# Example run
Sim = Monte_Carlo(
    ['AAPL', 'MSFT', 'SPY'],
    '2020-09-01',
    '2024-09-01',
    [1, 5, 6],
    0.05,
    10000,
    1,
    50
)

print(Sim.exploratory_analysis())
print("Historical VaR:", Sim.Historical_VaR())
print("Monte Carlo VaR, CVaR:", Sim.VaR_MC())
Sim.Visual()
print(Sim.back_test())
