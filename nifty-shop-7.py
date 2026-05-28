# # Cache fix for yfinance (add this BEFORE importing yfinance)
# from pathlib import Path
# import appdirs as ad
# CACHE_DIR = ".cache"
# ad.user_cache_dir = lambda *args: CACHE_DIR
# Path(CACHE_DIR).mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from scipy.optimize import brentq
import warnings
import io
warnings.filterwarnings('ignore')


# =============================================================================
# CORE ETFs FOR HIGHLIGHTING IN TRADE & POSITION TABLES
# These 10 ETFs form the "core" basket for long-term buy-average-hold.
# They are highlighted in yellow throughout the UI for visual prominence.
# =============================================================================
HIGHLIGHT_ETFS = {
    'NIFTYBEES', 'JUNIORBEES', 'MID150BEES', 'HDFCSML250', 'BANKBEES',
    'ITBEES', 'PHARMABEES', 'MON100', 'GOLDBEES', 'MOM30IETF'
}


def highlight_core_etfs(row):
    """Pandas Styler: highlight rows where Symbol is in the core 10-ETF list."""
    sym = str(row.get('symbol', row.get('Symbol', ''))).replace('.NS', '').strip()
    if sym in HIGHLIGHT_ETFS:
        return ['color: #fff; background-color: purple; font-weight: 600; color:#fff'] * len(row)
    return [''] * len(row)


# =============================================================================
# XIRR — true money-weighted CAGR from a list of dated cashflows
# Convention: outflows (investments) negative, inflows (sales / terminal MTM) positive
# =============================================================================
def xirr(cashflows, guess=0.1):
    """
    Compute XIRR via Brent's method.
    cashflows: list of (date, amount) tuples. Outflows negative, inflows positive.
    Returns annualized rate as a decimal (e.g. 0.1654 for 16.54%).
    Returns np.nan if it can't bracket a root.
    """
    if not cashflows or len(cashflows) < 2:
        return np.nan

    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    days = np.array([(d - t0).days for d, _ in cashflows], dtype=float)
    amts = np.array([a for _, a in cashflows], dtype=float)

    # Must have at least one positive and one negative cashflow
    if not (np.any(amts > 0) and np.any(amts < 0)):
        return np.nan

    def npv(rate):
        if rate <= -1.0:
            return np.inf
        return np.sum(amts / (1.0 + rate) ** (days / 365.25))

    # Bracket the root. Search broadly.
    lo, hi = -0.999, 10.0
    try:
        f_lo, f_hi = npv(lo), npv(hi)
        if np.isnan(f_lo) or np.isnan(f_hi):
            return np.nan
        if f_lo * f_hi > 0:
            # Try a wider negative bound first
            for try_hi in [50.0, 100.0]:
                f_hi = npv(try_hi)
                if f_lo * f_hi < 0:
                    hi = try_hi
                    break
            else:
                return np.nan
        return brentq(npv, lo, hi, xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan


class NiftyShopStrategy:
    """
    Implementation of the NIFTY SHOP Strategy as explained by FabTrader
    """
    def __init__(self, capital_per_trade=15000, target_percent=9.0, stop_loss_percent=9.0,
                 only_stocks=False, hold_core_etfs=True):
        self.capital_per_trade = capital_per_trade
        self.target_percent = target_percent / 100  # Convert to decimal
        self.stop_loss_percent = stop_loss_percent / 100  # Convert to decimal
        self.only_stocks = only_stocks
        # When True (default): core ETFs are never sold by the SELL logic — pure
        # accumulation. When False: core ETFs are signal-traded like any other
        # non-core name (target+EMA49 take-profit, stop-loss).
        self.hold_core_etfs = hold_core_etfs
        self.portfolio = {}
        self.cash = 0
        self.cash_deployed = 0
        self.trades = []
        self.equity_curve = []
        self.buy_signal_stocks = []

        # Fair benchmark tracking
        self.benchmark_portfolio = {}  # NIFTY positions bought when strategy trades
        self.benchmark_cash = 0
        self.benchmark_equity_curve = []
        self.benchmark_trades = []  # NEW: dated cashflow log for benchmark XIRR

        # Fair Core ETF benchmark: when the strategy deploys ₹X (BUY or AVERAGE),
        # we equally split ₹X across whichever core ETFs have data that day and
        # NEVER sell (pure buy-and-hold). Tests "would just holding the core
        # basket on the same deployment cadence have done better?"
        self.core_etf_portfolio = {}     # {symbol: {quantity, total_cost, first_buy_date}}
        self.core_etf_cash = 0
        self.core_etf_equity_curve = []
        self.core_etf_trades = []        # dated cashflow log for core-ETF XIRR

        self.core_etfs = [
            'NIFTYBEES.NS',   # Nifty 50 BeES - Highest liquidity benchmark ETF
            'JUNIORBEES.NS',  # NIFTY Junior BeES - Next 50 ETF
            'MID150BEES.NS',  # Midcap ETF
            'HDFCSML250.NS',  # HDFC Smallcap 250 ETF
            'BANKBEES.NS',    # Bank BeES - High liquidity sectoral ETF
            'ITBEES.NS',      # IT BeES - High liquidity tech ETF
            'PHARMABEES.NS',  # Pharma BeES
            # 'MAFANG.NS',    # FANG+ ETF
            'MON100.NS',      # Motilal Oswal NASDAQ 100 ETF
            'GOLDBEES.NS',    # Gold BeES - Commodity exposure
            'MOM30IETF.NS',   # MOM 30 ETF
        ]

        self.selected_etfs = self.core_etfs + [
            'SILVERBEES.NS',  # Silver BeES
            'SENSEXIETF.NS',  # Sensex ETF
            'BSE500IETF.NS',  # ICICI BSE 500 ETF
            'TOP100CASE.NS',
            'DIVOPPBEES.NS',  # Dividend Opportunities BeES
            'FMCGIETF.NS',    # FMCG ETF
            'TOP10ADD.NS',    # Top 10 ETF

            # Smart Beta / Factor-Based ETFs
            'SBIETFQLTY.NS',  # SBI Nifty 200 Quality 30 ETF
            'VAL30IETF.NS',   # Value 30 ETF
            'LOWVOLIETF.NS',  # Low Volatility 30 ETF
            # 'ALPHAETF.NS',
            'ALPL30IETF.NS',  # ALPL 30 ETF
            'HDFCGROWTH.NS',  # HDFC Growth ETF
            'HDFCQUAL.NS',    # HDFC Quality ETF
            'MULTICAP.NS',    # Multi Cap ETF

            # Sectoral/Thematic ETFs
            'MODEFENCE.NS',   # Mode Fence ETF
            'EVINDIA.NS',     # EV India ETF
            'MOREALTY.NS',    # Morealty ETF
            'AUTOBEES.NS',    # Auto BeES ETF
            'MOHEALTH.NS',    # MoHealth ETF
            'MAKEINDIA.NS',   # Make India ETF
            'PVTBANIETF.NS',  # PVT Bani ETF
            'SHARIABEES.NS',  # Sharia BeES ETF

            # International Diversification
            'HNGSNGBEES.NS'   # HNGS NGB BeES ETF
        ]

        self.stocks = [
            'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BHARTIARTL.NS',
            'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'ETERNAL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
            'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JIOFIN.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
            'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
            'IONEXCHANG.NS', 'DEEPAKNTR.NS', 'PERSISTENT.NS', 'KPITTECH.NS', 'AUBANK.NS', 'HAL.NS', 'NUVAMA.NS', 'WAAREEENER.NS', 'ASTRAL.NS', 'SONACOMS.NS', 'DIVISLAB.NS', 'CAMS.NS', 'POLYCAB.NS', 'PRAJIND.NS', 'CRISIL.NS', 'TATAPOWER.NS', 'DMART.NS', 
            'HAL.NS', 'MAZDOCK.NS', 'DIXON.NS', 'APLAPOLLO.NS', 'CDSL.NS', 'INDIGO.NS'
        ]

    def get_nifty50_data(self, start_date, end_date):
        """Download historical data for selected ETFs"""
        data = {}
        successful_downloads = []

        st.write("📊 Downloading ETF data...")
        progress_bar = st.progress(0)

        symbols = self.stocks if self.only_stocks else self.selected_etfs

        for i, symbol in enumerate(symbols):
            try:
                ticker_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not ticker_data.empty and len(ticker_data) > 25:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    data[symbol] = ticker_data
                    successful_downloads.append(symbol)
                progress_bar.progress((i + 1) / len(symbols))
            except Exception as e:
                st.warning(f"Failed to download {symbol}: {str(e)}")
                continue

        st.success(f"✅ Successfully downloaded data for {len(successful_downloads)} ETFs")
        return data, successful_downloads

    def get_benchmark_data(self, start_date, end_date):
        """Download NIFTY 50 benchmark data"""
        try:
            st.write("📊 Downloading NIFTY 50 benchmark data...")
            nifty_data = yf.download('NIFTYBEES.NS', start=start_date, end=end_date, progress=False)
            if isinstance(nifty_data.columns, pd.MultiIndex):
                nifty_data.columns = nifty_data.columns.droplevel(1)

            # Create CSV in memory
            csv_buffer = io.StringIO()
            nifty_data.to_csv(csv_buffer)

            # Add a download button for each symbol
            st.download_button(
                label=f"📥 Download NIFTYBEES CSV",
                data=csv_buffer.getvalue(),
                file_name=f"NIFTYBEES.csv",
                mime="text/csv"
            )

            return nifty_data
        except Exception as e:
            st.warning(f"Could not download NIFTY benchmark data: {e}")
            return pd.DataFrame()

    def calculate_signals(self, data):
        """Calculate buy/sell signals based on technical indicators"""
        signals = {}
        for symbol, df in data.items():
            try:
                df_copy = df.copy()
                if isinstance(df_copy.columns, pd.MultiIndex):
                    df_copy.columns = df_copy.columns.droplevel(1)
                if 'Close' not in df_copy.columns:
                    st.error(f"Close column not found for {symbol}")
                    continue
                # Calculate 9-day EMA
                df_copy['MA9'] = df_copy['Close'].ewm(span=9, adjust=False).mean()
                # Calculate 21-day EMA
                df_copy['MA21'] = df_copy['Close'].ewm(span=21, adjust=False).mean()
                # Calculate 21-day SMA
                df_copy['SMA21'] = df_copy['Close'].rolling(window=21, min_periods=1).mean()
                # Calculate 100-day SMA
                df_copy['SMA100'] = df_copy['Close'].rolling(window=100, min_periods=1).mean()
                # Calculate 49-day EMA
                df_copy['MA49'] = df_copy['Close'].ewm(span=49, adjust=False).mean()

                # Calculate distance from MA
                df_copy['Distance_from_MA'] = (df_copy['Close'] - df_copy['MA49']) / df_copy['MA49']
                # Calculate 52-week low
                df_copy['52w_low'] = df_copy['Close'].rolling(window=252, min_periods=20).min()
                df_copy['Dist_from_52w_low'] = (df_copy['Close'] - df_copy['52w_low']) / df_copy['52w_low']
                # Calculate RSI
                rsi = RSIIndicator(close=df_copy['Close'], window=14)
                df_copy['RSI_14'] = rsi.rsi()

                df_copy = df_copy.dropna()

                if not df_copy.empty:
                    signals[symbol] = df_copy
            except Exception as e:
                st.warning(f"Error processing {symbol}: {str(e)}")
                continue
        return signals

    def find_weakest_stocks(self, signals, date, exclude_held=None):
        """Find ETFs closest to their 52-week low"""
        if exclude_held is None:
            exclude_held = []
        distances = {}
        for symbol, df in signals.items():
            try:
                if date in df.index:
                    distance = df.loc[date, 'Dist_from_52w_low']
                    distances[symbol] = distance
            except Exception as e:
                continue
        sorted_stocks = sorted(distances.items(), key=lambda x: x[1])
        return [stock[0] for stock in sorted_stocks[:5]]

    def find_rsi_crossunder_35_stocks(self, signals, date):
        """Find stocks that cross under 35"""
        distances = {}
        for symbol, df in signals.items():
            try:
                if date in df.index:
                    rsi = df.loc[date, 'RSI_14']
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev_rsi = df.iloc[idx - 1]['RSI_14']
                    else:
                        prev_rsi = 35
                    if rsi < 35 and prev_rsi >= 35:
                        distances[symbol] = rsi
            except Exception as e:
                continue
        sorted_stocks = sorted(distances.items(), key=lambda x: x[1])
        return [stock[0] for stock in sorted_stocks]

    def find_buy_condition_matches(self, signals, symbol, date):
        """Find buy condition matches"""
        if symbol in signals and date in signals[symbol].index:
            try:
                price = signals[symbol].loc[date, 'Close']
                ema9 = signals[symbol].loc[date, 'MA9']
                ema21 = signals[symbol].loc[date, 'MA21']
                rsi = signals[symbol].loc[date, 'RSI_14']
                idx = signals[symbol].index.get_loc(date)
                if idx > 0:
                    prev_rsi = signals[symbol].iloc[idx - 1]['RSI_14']
                else:
                    prev_rsi = 35
                if idx > 1:
                    prev_rsi_2 = signals[symbol].iloc[idx - 2]['RSI_14']
                else:
                    prev_rsi_2 = 35
                if idx > 4:
                    prev_rsi_4 = signals[symbol].iloc[idx - 4]['RSI_14']
                else:
                    prev_rsi_4 = 35

                crossover_condition = prev_rsi <= prev_rsi_2 and rsi > prev_rsi

                if rsi > 35 and rsi < 70 and rsi > prev_rsi and prev_rsi_2 > prev_rsi_4 and price > ema21:
                    return True
            except Exception as e:
                print("Error", e)
                return False
        return False

    def update_benchmark(self, date, trade_amount, nifty_data, action='BUY'):
        """Update fair benchmark when strategy makes trades.
        Also logs each benchmark cashflow so we can compute a true XIRR later.
        """
        if nifty_data.empty or date not in nifty_data.index:
            return
        nifty_price = nifty_data.loc[date, 'Close']

        if action == 'BUY':
            # When strategy buys, benchmark buys equivalent NIFTY
            nifty_quantity = trade_amount / nifty_price
            self.benchmark_cash -= trade_amount

            if 'NIFTY' in self.benchmark_portfolio:
                self.benchmark_portfolio['NIFTY']['quantity'] += nifty_quantity
                total_cost = self.benchmark_portfolio['NIFTY']['total_cost'] + trade_amount
                total_quantity = self.benchmark_portfolio['NIFTY']['quantity']
                self.benchmark_portfolio['NIFTY']['avg_price'] = total_cost / total_quantity
                self.benchmark_portfolio['NIFTY']['total_cost'] = total_cost
            else:
                self.benchmark_portfolio['NIFTY'] = {
                    'quantity': nifty_quantity,
                    'avg_price': nifty_price,
                    'total_cost': trade_amount
                }

            # Log cashflow: outflow at this date
            self.benchmark_trades.append({
                'date': date, 'action': 'BUY', 'value': trade_amount
            })

        elif action == 'SELL' and 'NIFTY' in self.benchmark_portfolio:
            # When strategy sells, benchmark sells proportional NIFTY
            strategy_total_investment = sum([pos['total_cost'] for pos in self.portfolio.values()])
            if strategy_total_investment > 0:
                nifty_position = self.benchmark_portfolio['NIFTY']
                sell_proportion = trade_amount / (strategy_total_investment + trade_amount)
                quantity_to_sell = nifty_position['quantity'] * sell_proportion
                sell_value = quantity_to_sell * nifty_price

                self.benchmark_cash += sell_value
                self.benchmark_portfolio['NIFTY']['quantity'] -= quantity_to_sell
                self.benchmark_portfolio['NIFTY']['total_cost'] -= (sell_value * nifty_position['avg_price'] / nifty_price)

                if self.benchmark_portfolio['NIFTY']['quantity'] < 0.001:
                    del self.benchmark_portfolio['NIFTY']

                # Log cashflow: inflow at this date
                self.benchmark_trades.append({
                    'date': date, 'action': 'SELL', 'value': sell_value
                })

    def update_core_etf_benchmark(self, date, trade_amount, signals):
        """Fair Core-ETF benchmark — only fires on strategy BUYs (or AVERAGE).
        Splits `trade_amount` equally across whichever core ETFs have data
        on `date`. Never sells (core basket is buy-and-hold).
        Logs one cashflow entry per ETF for XIRR.
        """
        if trade_amount <= 0:
            return
        available = [e for e in self.core_etfs if e in signals and date in signals[e].index]
        if not available:
            return
        per_etf = trade_amount / len(available)
        self.core_etf_cash -= trade_amount
        for etf in available:
            try:
                px = float(signals[etf].loc[date, 'Close'])
                if px <= 0:
                    continue
                qty = per_etf / px
                pos = self.core_etf_portfolio.get(etf)
                if pos is None:
                    self.core_etf_portfolio[etf] = {
                        'quantity': qty, 'total_cost': per_etf, 'first_buy_date': date,
                    }
                else:
                    pos['quantity'] += qty
                    pos['total_cost'] += per_etf
                # Log cashflow: outflow at this date
                self.core_etf_trades.append({
                    'date': date, 'symbol': etf, 'action': 'BUY', 'value': per_etf,
                })
            except Exception:
                continue

    def backtest(self, start_date, end_date, initial_capital=1000000):
        """Run the complete backtest with fair benchmark"""
        # Get data
        data, successful_symbols = self.get_nifty50_data(start_date, end_date)
        nifty_data = self.get_benchmark_data(start_date, end_date)

        if not data:
            st.error("No data available for backtesting")
            return None, None, None

        # Calculate signals
        signals = self.calculate_signals(data)
        if not signals:
            st.error("No signals calculated - check data format")
            return None, None, None

        # Initialize
        self.cash = initial_capital
        self.benchmark_cash = initial_capital
        self.core_etf_cash = initial_capital
        self.portfolio = {}
        self.benchmark_portfolio = {}
        self.core_etf_portfolio = {}
        self.trades = []
        self.benchmark_trades = []
        self.core_etf_trades = []
        self.equity_curve = []
        self.benchmark_equity_curve = []
        self.core_etf_equity_curve = []

        # Get all trading dates
        all_dates = set()
        for df in signals.values():
            all_dates.update(df.index)
        trading_dates = sorted(list(all_dates))

        if len(trading_dates) < 21:
            st.error("Not enough trading days for backtesting")
            return None, None, None

        st.write("🔄 Running backtest with fair benchmark...")
        progress_bar = st.progress(0)

        for i, date in enumerate(trading_dates):
            if i < 20:
                continue

            # Calculate current portfolio values
            current_portfolio_value = self.cash
            benchmark_portfolio_value = self.benchmark_cash

            # Strategy portfolio value
            for symbol, position in self.portfolio.items():
                if symbol in signals and date in signals[symbol].index:
                    try:
                        current_price = signals[symbol].loc[date, 'Close']
                        current_portfolio_value += position['quantity'] * current_price
                    except Exception as e:
                        continue

            # Benchmark portfolio value
            if 'NIFTY' in self.benchmark_portfolio and date in nifty_data.index:
                try:
                    nifty_price = nifty_data.loc[date, 'Close']
                    benchmark_position = self.benchmark_portfolio['NIFTY']
                    benchmark_portfolio_value += benchmark_position['quantity'] * nifty_price
                except Exception as e:
                    pass

            # SELL LOGIC
            # When hold_core_etfs is True (default), core ETFs are skipped here —
            # they accumulate forever via BUY/AVERAGE and never exit.
            # When False, every position (core or non-core) is signal-traded.
            symbols_to_remove = []
            for symbol, position in self.portfolio.items():
                is_core = symbol in self.core_etfs
                if self.hold_core_etfs and is_core:
                    continue
                if symbol in signals and date in signals[symbol].index:
                    try:
                        current_price = signals[symbol].loc[date, 'Close']
                        profit_pct = (current_price - position['avg_price']) / position['avg_price']
                        ema49 = signals[symbol].loc[date, 'MA49']

                        if (profit_pct >= self.target_percent and current_price < ema49) or (profit_pct <= -self.stop_loss_percent):
                            sell_value = position['quantity'] * current_price
                            self.cash += sell_value
                            self.cash_deployed -= sell_value

                            self.update_benchmark(date, sell_value, nifty_data, 'SELL')

                            self.trades.append({
                                'date': date,
                                'first_buy_date': position['first_buy_date'],
                                'symbol': symbol,
                                'action': 'SELL',
                                'price': current_price,
                                'quantity': position['quantity'],
                                'value': sell_value,
                                'profit_pct': profit_pct * 100,
                                'profit_amount': sell_value - position['total_cost']
                            })
                            symbols_to_remove.append(symbol)
                    except Exception as e:
                        continue

            for symbol in symbols_to_remove:
                del self.portfolio[symbol]

            # BUY LOGIC
            held_symbols = list(self.portfolio.keys())
            rsi_crossunder_35_stocks = self.find_rsi_crossunder_35_stocks(signals, date)

            # NEW BUY Signal
            new_buys = 0
            if rsi_crossunder_35_stocks:
                for symbol in rsi_crossunder_35_stocks:
                    if date in signals[symbol].index:
                        try:
                            if symbol not in self.buy_signal_stocks:
                                self.buy_signal_stocks.append(symbol)
                            continue
                        except Exception as e:
                            continue

            # Actual buy
            averaging_stocks = {}
            for symbol in self.buy_signal_stocks:
                if date in signals[symbol].index:
                    buy_condition_matches = self.find_buy_condition_matches(signals, symbol, date)
                    if buy_condition_matches:
                        self.buy_signal_stocks.remove(symbol)
                        if symbol not in held_symbols and self.cash >= self.capital_per_trade:
                            price = signals[symbol].loc[date, 'Close']
                            quantity = int(self.capital_per_trade / price)
                            if quantity > 0:
                                cost = quantity * price
                                self.cash -= cost
                                self.cash_deployed += cost

                                self.update_benchmark(date, cost, nifty_data, 'BUY')
                                self.update_core_etf_benchmark(date, cost, signals)

                                self.portfolio[symbol] = {
                                    'quantity': quantity,
                                    'avg_price': price,
                                    'total_cost': cost,
                                    'last_buy_date': date,
                                    'last_buy_price': price,
                                    'average_count': 0,
                                    'first_buy_date': date
                                }

                                self.trades.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': price,
                                    'quantity': quantity,
                                    'value': cost,
                                    'profit_pct': 0,
                                    'profit_amount': 0
                                })
                                new_buys += 1
                        else:
                            distance = signals[symbol].loc[date, 'Dist_from_52w_low']
                            averaging_stocks[symbol] = distance

            # AVERAGING LOGIC
            if averaging_stocks:
                sorted_averaging_stocks = sorted(averaging_stocks.items(), key=lambda x: x[1])
                for symbol in [stock[0] for stock in sorted_averaging_stocks]:
                    if symbol in signals and date in signals[symbol].index:
                        try:
                            current_price = signals[symbol].loc[date, 'Close']
                            quantity = int(self.capital_per_trade / current_price)
                            old_position = self.portfolio[symbol]
                            if self.cash >= self.capital_per_trade and quantity > 0 and old_position['average_count'] < 25:
                                cost = quantity * current_price
                                self.cash -= cost
                                self.cash_deployed += cost

                                self.update_benchmark(date, cost, nifty_data, 'BUY')
                                self.update_core_etf_benchmark(date, cost, signals)

                                new_total_quantity = old_position['quantity'] + quantity
                                new_total_cost = old_position['total_cost'] + cost
                                new_avg_price = new_total_cost / new_total_quantity
                                new_average_count = old_position['average_count'] + 1

                                self.portfolio[symbol] = {
                                    'quantity': new_total_quantity,
                                    'avg_price': new_avg_price,
                                    'total_cost': new_total_cost,
                                    'last_buy_date': date,
                                    'last_buy_price': current_price,
                                    'average_count': new_average_count,
                                    'first_buy_date': old_position['first_buy_date']
                                }

                                self.trades.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'action': 'AVERAGE',
                                    'price': current_price,
                                    'quantity': quantity,
                                    'value': cost,
                                    'profit_pct': 0,
                                    'profit_amount': 0
                                })
                        except Exception as e:
                            continue

            # Final portfolio values for equity curve
            final_portfolio_value = self.cash
            final_benchmark_value = self.benchmark_cash
            final_core_etf_value = self.core_etf_cash

            for symbol, position in self.portfolio.items():
                if symbol in signals and date in signals[symbol].index:
                    try:
                        current_price = signals[symbol].loc[date, 'Close']
                        final_portfolio_value += position['quantity'] * current_price
                    except Exception as e:
                        continue

            if 'NIFTY' in self.benchmark_portfolio and date in nifty_data.index:
                try:
                    nifty_price = nifty_data.loc[date, 'Close']
                    benchmark_position = self.benchmark_portfolio['NIFTY']
                    final_benchmark_value += benchmark_position['quantity'] * nifty_price
                except Exception as e:
                    pass

            # Mark-to-market the Fair Core ETF basket
            for etf, position in self.core_etf_portfolio.items():
                if etf in signals and date in signals[etf].index:
                    try:
                        px = signals[etf].loc[date, 'Close']
                        final_core_etf_value += position['quantity'] * px
                    except Exception:
                        final_core_etf_value += position['total_cost']

            self.equity_curve.append({
                'date': date,
                'portfolio_value': final_portfolio_value,
                'cash': self.cash,
                'positions': len(self.portfolio)
            })

            self.benchmark_equity_curve.append({
                'date': date,
                'portfolio_value': final_benchmark_value,
                'cash': self.benchmark_cash,
                'positions': 1 if 'NIFTY' in self.benchmark_portfolio else 0
            })

            self.core_etf_equity_curve.append({
                'date': date,
                'portfolio_value': final_core_etf_value,
                'cash': self.core_etf_cash,
                'positions': len(self.core_etf_portfolio),
            })

            progress_bar.progress((i + 1) / len(trading_dates))

        self.last_signals = signals
        return (
            pd.DataFrame(self.equity_curve),
            pd.DataFrame(self.benchmark_equity_curve),
            pd.DataFrame(self.core_etf_equity_curve),
        )

    # =========================================================================
    # CORRECTED RETURN METRICS
    # =========================================================================
    def _build_strategy_cashflows(self, equity_curve):
        """Build strategy cashflows from trade log + terminal MTM of open positions.
        Each BUY/AVERAGE is an outflow, each SELL is an inflow,
        and the final mark-to-market value of all still-open positions
        is added as a terminal inflow on the last date.
        Returns: (cashflows list, total_invested, total_received, terminal_mtm)
        """
        cashflows = []
        total_invested = 0.0
        total_received = 0.0

        for t in self.trades:
            if t['action'] in ('BUY', 'AVERAGE'):
                cashflows.append((pd.Timestamp(t['date']), -float(t['value'])))
                total_invested += float(t['value'])
            elif t['action'] == 'SELL':
                cashflows.append((pd.Timestamp(t['date']), float(t['value'])))
                total_received += float(t['value'])

        # Terminal mark-to-market: value of still-open positions on last date
        terminal_mtm = 0.0
        if equity_curve is not None and not equity_curve.empty:
            last_date = pd.Timestamp(equity_curve['date'].iloc[-1])
            for symbol, position in self.portfolio.items():
                try:
                    if hasattr(self, 'last_signals') and symbol in self.last_signals:
                        sig = self.last_signals[symbol]
                        if last_date in sig.index:
                            px = sig.loc[last_date, 'Close']
                        else:
                            px = sig['Close'].iloc[-1]
                    else:
                        px = position['avg_price']
                    terminal_mtm += position['quantity'] * px
                except Exception:
                    terminal_mtm += position['total_cost']
            if terminal_mtm > 0:
                cashflows.append((last_date, float(terminal_mtm)))

        return cashflows, total_invested, total_received, terminal_mtm

    def _build_benchmark_cashflows(self, benchmark_curve, nifty_data):
        """Build benchmark cashflows from the logged benchmark trades + terminal MTM."""
        cashflows = []
        total_invested = 0.0
        total_received = 0.0

        for t in self.benchmark_trades:
            if t['action'] == 'BUY':
                cashflows.append((pd.Timestamp(t['date']), -float(t['value'])))
                total_invested += float(t['value'])
            elif t['action'] == 'SELL':
                cashflows.append((pd.Timestamp(t['date']), float(t['value'])))
                total_received += float(t['value'])

        terminal_mtm = 0.0
        if benchmark_curve is not None and not benchmark_curve.empty:
            last_date = pd.Timestamp(benchmark_curve['date'].iloc[-1])
            if 'NIFTY' in self.benchmark_portfolio:
                try:
                    if last_date in nifty_data.index:
                        nifty_px = nifty_data.loc[last_date, 'Close']
                    else:
                        nifty_px = nifty_data['Close'].iloc[-1]
                    terminal_mtm = self.benchmark_portfolio['NIFTY']['quantity'] * nifty_px
                except Exception:
                    terminal_mtm = self.benchmark_portfolio['NIFTY']['total_cost']
                if terminal_mtm > 0:
                    cashflows.append((last_date, float(terminal_mtm)))

        return cashflows, total_invested, total_received, terminal_mtm

    def _build_core_etf_cashflows(self, core_curve, signals):
        """Build Fair Core ETF benchmark cashflows from logged BUYs + terminal MTM.
        Core ETFs are buy-and-hold, so there are no SELL inflows — only the
        terminal mark-to-market of all open ETF positions at the last date.
        """
        cashflows = []
        total_invested = 0.0
        total_received = 0.0

        for t in self.core_etf_trades:
            # All entries are BUYs (buy-and-hold design)
            cashflows.append((pd.Timestamp(t['date']), -float(t['value'])))
            total_invested += float(t['value'])

        terminal_mtm = 0.0
        if core_curve is not None and not core_curve.empty and self.core_etf_portfolio:
            last_date = pd.Timestamp(core_curve['date'].iloc[-1])
            for etf, position in self.core_etf_portfolio.items():
                try:
                    if etf in signals and last_date in signals[etf].index:
                        px = float(signals[etf].loc[last_date, 'Close'])
                    elif etf in signals:
                        px = float(signals[etf]['Close'].iloc[-1])
                    else:
                        px = position['total_cost'] / position['quantity'] if position['quantity'] else 0
                    terminal_mtm += position['quantity'] * px
                except Exception:
                    terminal_mtm += position['total_cost']
            if terminal_mtm > 0:
                cashflows.append((last_date, float(terminal_mtm)))

        return cashflows, total_invested, total_received, terminal_mtm

    @staticmethod
    def _capital_weighted_years(cashflows, terminal_date):
        """Compute capital-weighted holding period (in years) using only the outflows.
        Used for 'CAGR on deployed capital' — duration weighted by ₹ invested at each date.
        """
        outflows = [(d, -a) for d, a in cashflows if a < 0]
        if not outflows:
            return 0.0
        td = pd.Timestamp(terminal_date)
        total_w = sum(amt for _, amt in outflows)
        if total_w <= 0:
            return 0.0
        weighted_days = sum((td - d).days * amt for d, amt in outflows) / total_w
        return weighted_days / 365.25

    def calculate_metrics(self, equity_curve, initial_capital, nifty_data=None):
        """Calculate strategy performance metrics with CORRECTED XIRR and deployed-capital CAGR.

        Three return measures reported:
          1. Total Return on initial capital — full portfolio value vs starting cash
          2. CAGR on deployed capital — annualized on actual ₹ put to work, using
             capital-weighted holding period. This is what the user asked for.
          3. XIRR — true money-weighted IRR over every trade-level cashflow.
        """
        if equity_curve is None or equity_curve.empty:
            return {}

        final_value = equity_curve['portfolio_value'].iloc[-1]
        total_return_on_capital = (final_value - initial_capital) / initial_capital

        # Build cashflows
        cashflows, total_invested, total_received, terminal_mtm = self._build_strategy_cashflows(equity_curve)
        last_date = equity_curve['date'].iloc[-1]

        # ---- CAGR on deployed capital ----
        net_value = total_received + terminal_mtm  # everything that came back (realized + unrealized)
        if total_invested > 0 and net_value > 0:
            wtd_years = self._capital_weighted_years(cashflows, last_date)
            if wtd_years > 0:
                cagr_deployed = (net_value / total_invested) ** (1 / wtd_years) - 1
            else:
                cagr_deployed = np.nan
        else:
            wtd_years = 0.0
            cagr_deployed = np.nan

        # ---- XIRR (money-weighted) ----
        xirr_value = xirr(cashflows)

        # ---- Win rate ----
        profitable_trades = [t for t in self.trades if t['action'] == 'SELL' and t['profit_amount'] > 0]
        total_sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        win_rate = len(profitable_trades) / len(total_sell_trades) if total_sell_trades else 0

        # ---- Max drawdown ----
        ec = equity_curve.copy()
        ec['rolling_max'] = ec['portfolio_value'].expanding().max()
        ec['drawdown'] = (ec['portfolio_value'] - ec['rolling_max']) / ec['rolling_max']
        max_drawdown = ec['drawdown'].min()

        return {
            'Total Return': f"{total_return_on_capital:.2%}",
            'CAGR (deployed)': f"{cagr_deployed:.2%}" if not np.isnan(cagr_deployed) else "N/A",
            'XIRR': f"{xirr_value:.2%}" if not np.isnan(xirr_value) else "N/A",
            'Final Value': f"₹{final_value:,.0f}",
            'Total Invested': f"₹{total_invested:,.0f}",
            'Capital-Wtd Years': f"{wtd_years:.2f}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Trades': len(total_sell_trades),
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Current Positions': len(self.portfolio),
            # raw numeric versions for downstream comparisons
            '_total_return_num': total_return_on_capital * 100,
            '_cagr_deployed_num': (cagr_deployed * 100) if not np.isnan(cagr_deployed) else np.nan,
            '_xirr_num': (xirr_value * 100) if not np.isnan(xirr_value) else np.nan,
        }

    def calculate_benchmark_metrics(self, benchmark_curve, initial_capital, nifty_data):
        """Calculate benchmark (NIFTY 50) metrics on the same trade-event cashflows."""
        if benchmark_curve is None or benchmark_curve.empty:
            return {}

        final_value = benchmark_curve['portfolio_value'].iloc[-1]
        total_return_on_capital = (final_value - initial_capital) / initial_capital

        cashflows, total_invested, total_received, terminal_mtm = self._build_benchmark_cashflows(benchmark_curve, nifty_data)
        last_date = benchmark_curve['date'].iloc[-1]

        net_value = total_received + terminal_mtm
        if total_invested > 0 and net_value > 0:
            wtd_years = self._capital_weighted_years(cashflows, last_date)
            cagr_deployed = (net_value / total_invested) ** (1 / wtd_years) - 1 if wtd_years > 0 else np.nan
        else:
            wtd_years = 0.0
            cagr_deployed = np.nan

        xirr_value = xirr(cashflows)

        return {
            'Total Return': f"{total_return_on_capital:.2%}",
            'CAGR (deployed)': f"{cagr_deployed:.2%}" if not np.isnan(cagr_deployed) else "N/A",
            'XIRR': f"{xirr_value:.2%}" if not np.isnan(xirr_value) else "N/A",
            'Final Value': f"₹{final_value:,.0f}",
            'Total Invested': f"₹{total_invested:,.0f}",
            'Capital-Wtd Years': f"{wtd_years:.2f}",
            '_total_return_num': total_return_on_capital * 100,
            '_cagr_deployed_num': (cagr_deployed * 100) if not np.isnan(cagr_deployed) else np.nan,
            '_xirr_num': (xirr_value * 100) if not np.isnan(xirr_value) else np.nan,
        }

    def calculate_core_etf_metrics(self, core_curve, initial_capital, signals):
        """Calculate Fair Core ETF benchmark metrics — buy-and-hold equal-weight
        across whichever core ETFs were available when the strategy deployed."""
        if core_curve is None or core_curve.empty:
            return {}

        final_value = core_curve['portfolio_value'].iloc[-1]
        total_return_on_capital = (final_value - initial_capital) / initial_capital

        cashflows, total_invested, total_received, terminal_mtm = self._build_core_etf_cashflows(core_curve, signals)
        last_date = core_curve['date'].iloc[-1]

        net_value = total_received + terminal_mtm
        if total_invested > 0 and net_value > 0:
            wtd_years = self._capital_weighted_years(cashflows, last_date)
            cagr_deployed = (net_value / total_invested) ** (1 / wtd_years) - 1 if wtd_years > 0 else np.nan
        else:
            wtd_years = 0.0
            cagr_deployed = np.nan

        xirr_value = xirr(cashflows)

        return {
            'Total Return': f"{total_return_on_capital:.2%}",
            'CAGR (deployed)': f"{cagr_deployed:.2%}" if not np.isnan(cagr_deployed) else "N/A",
            'XIRR': f"{xirr_value:.2%}" if not np.isnan(xirr_value) else "N/A",
            'Final Value': f"₹{final_value:,.0f}",
            'Total Invested': f"₹{total_invested:,.0f}",
            'Capital-Wtd Years': f"{wtd_years:.2f}",
            'Open Positions': f"{len(self.core_etf_portfolio)}",
            '_total_return_num': total_return_on_capital * 100,
            '_cagr_deployed_num': (cagr_deployed * 100) if not np.isnan(cagr_deployed) else np.nan,
            '_xirr_num': (xirr_value * 100) if not np.isnan(xirr_value) else np.nan,
        }


def main():
    st.set_page_config(page_title="NIFTY SHOP Strategy - Dip-Buy, Average & Hold", layout="wide")
    st.title("🏪 NIFTY SHOP Strategy - Dip-Buy, Average & Hold")
    st.markdown("**Based on NIFTY ETF Shop Strategy**")

    # Sidebar for parameters
    st.sidebar.header("Strategy Parameters")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=1097))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=2000000, step=10000)
    capital_per_trade = st.sidebar.number_input("Capital per Trade (₹)", value=10000, step=1000)
    target_percent = st.sidebar.number_input("Target Profit (%)", value=9.00, step=0.1)
    stop_loss_percent = st.sidebar.number_input("Stop Loss (%)", value=1000.00, step=0.1)
    only_stocks = st.sidebar.checkbox("Only Stocks", value=False)
    hold_core_etfs = st.sidebar.checkbox(
        "Hold core ETFs (never sell)",
        value=True,
        help="When ON (default): the 12 core ETFs (NIFTYBEES, JUNIORBEES, MID150BEES, "
             "HDFCSML250, BANKBEES, ITBEES, PHARMABEES, MON100, GOLDBEES, SILVERBEES, "
             "SENSEXIETF, BSE500IETF) accumulate forever — no take-profit, no stop-loss. "
             "When OFF: core ETFs are traded like any other position (9% take-profit on "
             "weakness, stop-loss if configured).",
    )

    # Highlight legend in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Highlighted Core ETFs**")
    st.sidebar.caption("These 10 ETFs are highlighted in trade & position tables:")
    st.sidebar.caption(", ".join(HIGHLIGHT_ETFS))

    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            strategy = NiftyShopStrategy(
                capital_per_trade=capital_per_trade,
                target_percent=target_percent,
                stop_loss_percent=stop_loss_percent,
                only_stocks=only_stocks,
                hold_core_etfs=hold_core_etfs,
            )

            equity_curve, benchmark_curve, core_etf_curve = strategy.backtest(start_date, end_date, initial_capital)

            if equity_curve is not None and not equity_curve.empty:
                # Need nifty_data for benchmark cashflows — re-fetch (it's cached by yfinance)
                nifty_data = yf.download('NIFTYBEES.NS', start=start_date, end=end_date, progress=False)
                if isinstance(nifty_data.columns, pd.MultiIndex):
                    nifty_data.columns = nifty_data.columns.droplevel(1)

                strategy_metrics = strategy.calculate_metrics(equity_curve, initial_capital, nifty_data)
                benchmark_metrics = strategy.calculate_benchmark_metrics(benchmark_curve, initial_capital, nifty_data)
                core_etf_metrics = strategy.calculate_core_etf_metrics(
                    core_etf_curve, initial_capital,
                    strategy.last_signals if hasattr(strategy, 'last_signals') else {},
                )

                # =================================================================
                # PERFORMANCE COMPARISON
                # =================================================================
                st.header("📊 Performance Comparison")
                st.caption(
                    "**CAGR (deployed)** annualizes returns over actual capital-weighted holding period — "
                    "the right number when capital is deployed gradually. "
                    "**XIRR** is the money-weighted IRR over every individual cashflow. "
                    "**Fair Core ETFs** mirrors every strategy BUY/AVERAGE by equally splitting that "
                    "₹ amount across whichever of the 12 core ETFs have data that day — and never sells. "
                    "Tests whether the dip-buy signal beats simple diversified accumulation."
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.subheader("🎯 Strategy")
                    st.metric("Total Return", strategy_metrics['Total Return'])
                    st.metric("XIRR", strategy_metrics['XIRR'])
                    st.metric("CAGR (deployed)", strategy_metrics['CAGR (deployed)'])
                    st.metric("Final Value", strategy_metrics['Final Value'])
                    st.metric("Total Invested", strategy_metrics['Total Invested'])
                    st.metric("Capital-Wtd Years", strategy_metrics['Capital-Wtd Years'])
                    st.metric("Win Rate", strategy_metrics['Win Rate'])
                    st.metric("Max Drawdown", strategy_metrics['Max Drawdown'])

                with col2:
                    st.subheader("📈 Fair NIFTY 50")
                    if benchmark_metrics:
                        st.metric("Total Return", benchmark_metrics['Total Return'])
                        st.metric("XIRR", benchmark_metrics['XIRR'])
                        st.metric("CAGR (deployed)", benchmark_metrics['CAGR (deployed)'])
                        st.metric("Final Value", benchmark_metrics['Final Value'])
                        st.metric("Total Invested", benchmark_metrics['Total Invested'])
                        st.metric("Capital-Wtd Years", benchmark_metrics['Capital-Wtd Years'])
                    else:
                        st.info("Benchmark data unavailable")

                with col3:
                    st.subheader("🟢 Fair Core ETFs")
                    if core_etf_metrics:
                        st.metric("Total Return", core_etf_metrics['Total Return'])
                        st.metric("XIRR", core_etf_metrics['XIRR'])
                        st.metric("CAGR (deployed)", core_etf_metrics['CAGR (deployed)'])
                        st.metric("Final Value", core_etf_metrics['Final Value'])
                        st.metric("Total Invested", core_etf_metrics['Total Invested'])
                        st.metric("Capital-Wtd Years", core_etf_metrics['Capital-Wtd Years'])
                        st.metric("Open Positions", core_etf_metrics['Open Positions'])
                    else:
                        st.info("Core ETF benchmark unavailable")

                with col4:
                    st.subheader("🏆 Strategy vs Both")
                    if benchmark_metrics:
                        ex_xirr_n = strategy_metrics['_xirr_num'] - benchmark_metrics['_xirr_num'] if not (np.isnan(strategy_metrics['_xirr_num']) or np.isnan(benchmark_metrics['_xirr_num'])) else np.nan
                        ex_cagr_n = strategy_metrics['_cagr_deployed_num'] - benchmark_metrics['_cagr_deployed_num'] if not (np.isnan(strategy_metrics['_cagr_deployed_num']) or np.isnan(benchmark_metrics['_cagr_deployed_num'])) else np.nan
                        st.metric("Excess XIRR vs N50",
                                  f"{ex_xirr_n:.2f}%" if not np.isnan(ex_xirr_n) else "N/A",
                                  f"{ex_xirr_n:.2f}%" if not np.isnan(ex_xirr_n) else None)
                        st.metric("Excess CAGR vs N50",
                                  f"{ex_cagr_n:.2f}%" if not np.isnan(ex_cagr_n) else "N/A",
                                  f"{ex_cagr_n:.2f}%" if not np.isnan(ex_cagr_n) else None)
                    if core_etf_metrics:
                        ex_xirr_c = strategy_metrics['_xirr_num'] - core_etf_metrics['_xirr_num'] if not (np.isnan(strategy_metrics['_xirr_num']) or np.isnan(core_etf_metrics['_xirr_num'])) else np.nan
                        ex_cagr_c = strategy_metrics['_cagr_deployed_num'] - core_etf_metrics['_cagr_deployed_num'] if not (np.isnan(strategy_metrics['_cagr_deployed_num']) or np.isnan(core_etf_metrics['_cagr_deployed_num'])) else np.nan
                        st.metric("Excess XIRR vs Core ETFs",
                                  f"{ex_xirr_c:.2f}%" if not np.isnan(ex_xirr_c) else "N/A",
                                  f"{ex_xirr_c:.2f}%" if not np.isnan(ex_xirr_c) else None)
                        st.metric("Excess CAGR vs Core ETFs",
                                  f"{ex_cagr_c:.2f}%" if not np.isnan(ex_cagr_c) else "N/A",
                                  f"{ex_cagr_c:.2f}%" if not np.isnan(ex_cagr_c) else None)
                    st.metric("Total Trades", strategy_metrics['Total Trades'])
                    st.metric("Current Positions", strategy_metrics['Current Positions'])

                # =================================================================
                # EQUITY CURVES
                # =================================================================
                st.header("📈 Strategy vs Fair Benchmark Performance")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_curve['date'], y=equity_curve['portfolio_value'],
                    mode='lines', name='NIFTY SHOP Strategy',
                    line=dict(color='#1f77b4', width=3)
                ))
                if not benchmark_curve.empty:
                    fig.add_trace(go.Scatter(
                        x=benchmark_curve['date'], y=benchmark_curve['portfolio_value'],
                        mode='lines', name='Fair NIFTY 50 Benchmark',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                fig.update_layout(
                    title="Portfolio Value Over Time - Strategy vs Fair Benchmark",
                    xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
                    hovermode='x unified', height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Capital deployment
                st.header("💰 Capital Deployment Comparison")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=equity_curve['date'], y=equity_curve['cash'],
                    mode='lines', name='Strategy Cash', line=dict(color='#2ca02c', width=2)
                ))
                if not benchmark_curve.empty:
                    fig2.add_trace(go.Scatter(
                        x=benchmark_curve['date'], y=benchmark_curve['cash'],
                        mode='lines', name='Benchmark Cash', line=dict(color='#d62728', width=2)
                    ))
                fig2.update_layout(
                    title="Cash Levels Over Time - Showing Fair Capital Deployment",
                    xaxis_title="Date", yaxis_title="Cash (₹)",
                    hovermode='x unified', height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

                # =================================================================
                # RECENT TRADES — with core ETF highlighting
                # =================================================================
                st.header("📋 Recent Trades")
                st.caption("Purple rows = Core 10 ETFs (buy-average-hold candidates)")

                if strategy.trades:
                    trades_df = pd.DataFrame(strategy.trades)
                    trades_df = trades_df.sort_values('date', ascending=False).reset_index(drop=True)

                    display_df = trades_df.copy()
                    display_df['value'] = display_df['value'].apply(lambda x: f"₹{x:,.0f}")
                    display_df['price'] = display_df['price'].apply(lambda x: f"₹{x:.2f}")
                    display_df['profit_pct'] = display_df['profit_pct'].apply(lambda x: f"{x:.2f}%")
                    display_df['profit_amount'] = display_df['profit_amount'].apply(lambda x: f"₹{x:,.0f}")

                    styled = display_df.style.apply(highlight_core_etfs, axis=1)
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.info("No trades executed during the backtest period")

                # =================================================================
                # CURRENT POSITIONS — with core ETF highlighting
                # =================================================================
                st.header("💼 Current Positions")
                st.caption("Purple rows = Core 10 ETFs (buy-average-hold candidates)")

                if strategy.portfolio:
                    positions_data = []
                    last_date = equity_curve['date'].iloc[-1] if not equity_curve.empty else None

                    for symbol, position in strategy.portfolio.items():
                        try:
                            if hasattr(strategy, 'last_signals') and symbol in strategy.last_signals:
                                signals_data = strategy.last_signals[symbol]
                                if last_date in signals_data.index:
                                    current_price = signals_data.loc[last_date, 'Close']
                                else:
                                    current_price = signals_data['Close'].iloc[-1]
                            else:
                                try:
                                    current_data = yf.download(symbol, period="5d", progress=False)
                                    if not current_data.empty:
                                        if isinstance(current_data.columns, pd.MultiIndex):
                                            current_data.columns = current_data.columns.droplevel(1)
                                        current_price = current_data['Close'].iloc[-1]
                                    else:
                                        current_price = position['avg_price']
                                except:
                                    current_price = position['avg_price']

                            current_value = position['quantity'] * current_price
                            unrealized_pnl = current_value - position['total_cost']
                            unrealized_pnl_pct = (unrealized_pnl / position['total_cost']) * 100

                            positions_data.append({
                                'Symbol': symbol.replace('.NS', ''),
                                'First Buy Date': position['first_buy_date'],
                                'Quantity': position['quantity'],
                                'Avg Price': f"₹{position['avg_price']:.2f}",
                                'Current Price': f"₹{current_price:.2f}",
                                'Investment': f"₹{position['total_cost']:,.0f}",
                                'Current Value': f"₹{current_value:,.0f}",
                                'Unrealized P&L': f"₹{unrealized_pnl:,.0f}",
                                'P&L %': f"{unrealized_pnl_pct:.2f}%",
                                'Avg Count': position['average_count']
                            })
                        except Exception as e:
                            st.warning(f"Could not process position for {symbol}: {str(e)}")
                            continue

                    if positions_data:
                        positions_df = pd.DataFrame(positions_data)
                        styled_pos = positions_df.style.apply(highlight_core_etfs, axis=1)
                        st.dataframe(styled_pos, use_container_width=True)

                        total_investment = sum([pos['total_cost'] for pos in strategy.portfolio.values()])
                        # Count of core-ETF positions
                        core_held = sum(1 for sym in strategy.portfolio.keys()
                                        if sym.replace('.NS', '') in HIGHLIGHT_ETFS)
                        st.info(
                            f"**Total Positions:** {len(positions_data)} | "
                            f"**Core ETFs Held:** {core_held}/{len(HIGHLIGHT_ETFS)} | "
                            f"**Total Investment:** ₹{total_investment:,.0f}"
                        )
                    else:
                        st.warning("Could not display position details due to data issues")
                else:
                    st.info("No current positions")

                # Buy Signal Symbol
                st.header("Buy Signal Symbol")
                if strategy.buy_signal_stocks:
                    st.write(strategy.buy_signal_stocks)
                else:
                    st.info("No buy signals")
            else:
                st.error("Backtesting failed. Please check your parameters and try again.")


if __name__ == "__main__":
    main()
