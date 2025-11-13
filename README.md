# Monte Carlo VaR Simulator

This project demonstrates a **Monte Carlo-based Value-at-Risk (VaR) and Conditional VaR (CVaR)** calculation for a multi-asset portfolio using **Python**.  
It integrates **historical simulation**, **Monte Carlo simulation**, and **statistical analysis** for portfolio risk estimation.

---

## Key Features
- Downloads historical price data using `yfinance`
- Computes log returns and performs exploratory data analysis
- Estimates **Historical VaR** and **Monte Carlo VaR / CVaR**
- Visualizes simulated vs. historical return distributions
- Performs simple **backtesting** for VaR exceedances

---

## Tech Stack
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- SciPy
- yFinance

---

## Installation

```bash
git clone https://github.com/<your-username>/monte-carlo-var-simulator.git
cd monte-carlo-var-simulator
pip install -r requirements.txt
