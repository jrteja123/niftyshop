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
import warnings
import io

warnings.filterwarnings('ignore')

class NiftyShopStrategy:
    """
    Implementation of the NIFTY SHOP Strategy as explained by FabTrader
    """
    
    def __init__(self, capital_per_trade=15000, target_percent=9.0, stop_loss_percent=9.0):
        self.capital_per_trade = capital_per_trade
        self.target_percent = target_percent / 100  # Convert to decimal
        self.stop_loss_percent = stop_loss_percent / 100  # Convert to decimal
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
        self.core_etfs = [
            'NIFTYBEES.NS',      # Nifty 50 BeES - Highest liquidity benchmark ETF
            'JUNIORBEES.NS',     # NIFTY Junior BeES - Next 50 ETF
            'MID150BEES.NS',     # Midcap ETF
            'HDFCSML250.NS',     # HDFC Smallcap 250 ETF

            'BANKBEES.NS',       # Bank BeES - High liquidity sectoral ETF
            'ITBEES.NS',         # IT BeES - High liquidity tech ETF
            'PHARMABEES.NS',     # Pharma BeES

            'DIVOPPBEES.NS',     # Dividend Opportunities BeES
            'FMCGIETF.NS',       # FMCG ETF
            'TOP10ADD.NS',       # Top 10 ETF

            #'MAFANG.NS',         # FANG+ ETF
            'MON100.NS',         # Motilal Oswal NASDAQ 100 ETF

            'GOLDBEES.NS',       # Gold BeES - Commodity exposure
            'SILVERBEES.NS',     # Silver BeES
            
            'SENSEXIETF.NS',     # Sensex ETF
            'BSE500IETF.NS',     # ICICI BSE 500 ETF
        ]

        self.selected_etfs = self.core_etfs + [
            'TOP100CASE.NS',

            # Smart Beta / Factor-Based ETFs
            'SBIETFQLTY.NS', #SBI Nifty 200 Quality 30 ETF
            'MOM30IETF.NS', #MOM 30 ETF
            'VAL30IETF.NS', #Value 30 ETF
            'LOWVOLIETF.NS', #Low Volatility 30 ETF
            #'ALPHAETF.NS',
            'ALPL30IETF.NS', #ALPL 30 ETF
            'HDFCGROWTH.NS', #HDFC Growth ETF
            'HDFCQUAL.NS', #HDFC Quality ETF
            'MULTICAP.NS', #Multi Cap ETF
            
            # Sectoral/Thematic ETFs
            'MODEFENCE.NS', #Mode Fence ETF
            'EVINDIA.NS', #EV India ETF
            'MOREALTY.NS', #Morealty ETF
            'AUTOBEES.NS', #Auto BeES ETF
            'MOHEALTH.NS', #MoHealth ETF
            'MAKEINDIA.NS', #Make India ETF
            'PVTBANIETF.NS', #PVT Bani ETF
            'SHARIABEES.NS', #Sharia BeES ETF

            # International Diversification
            'HNGSNGBEES.NS' #HNGS NGB BeES ETF
        ]

    def get_nifty50_data(self, start_date, end_date):
        """Download historical data for selected ETFs"""
        data = {}
        successful_downloads = []
        
        st.write("📊 Downloading ETF data...")
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(self.selected_etfs):
            try:
                ticker_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not ticker_data.empty and len(ticker_data) > 25:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    data[symbol] = ticker_data
                    successful_downloads.append(symbol)
                progress_bar.progress((i + 1) / len(self.selected_etfs))
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

                # Calculate 34-day EMA
                df_copy['SMA21'] = df_copy['Close'].rolling(window=21, min_periods=1).mean()

                # Calculate 49-day EMA
                df_copy['MA49'] = df_copy['Close'].ewm(span=49, adjust=False).mean()

                # # Calculate 54-day EMA
                # df_copy['MA54'] = df_copy['Close'].ewm(span=54, adjust=False).mean()
                
                # Calculate distance from MA
                df_copy['Distance_from_MA'] = (df_copy['Close'] - df_copy['MA49']) / df_copy['MA49']
                
                # Calculate 52-week low
                df_copy['52w_low'] = df_copy['Close'].rolling(window=252, min_periods=20).min()
                df_copy['Dist_from_52w_low'] = (df_copy['Close'] - df_copy['52w_low']) / df_copy['52w_low']
                
                # Calculate RSI
                rsi = RSIIndicator(close=df_copy['Close'], window=14)
                df_copy['RSI_14'] = rsi.rsi()

                # # Calculate xsignals
                # crossunder_35 = ta.xsignals(df_copy['RSI_14'], 35, 70, above=False)
                # print(crossunder_35)
                # df_copy['crossunder_35'] = crossunder_35

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

                #"""condtion -  rsi > 35 and rsi < 70 and ta.crossover(rsi, rsi[1]) and rsi[2] > rsi[4] and close > ema9 and sma100 > close"""
                # if symbol == 'AUTOBEES.NS':
                #     print("Inside", date, symbol, rsi, prev_rsi, price, ema9)

                if rsi > 35 and rsi < 70 and rsi > prev_rsi and price > ema9 and prev_rsi_2 > prev_rsi_4:
                    return True
            except Exception as e:
                print("Error", e)
                return False
        return False
    
    def update_benchmark(self, date, trade_amount, nifty_data, action='BUY'):
        """Update fair benchmark when strategy makes trades"""
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
        
        elif action == 'SELL' and 'NIFTY' in self.benchmark_portfolio:
            # When strategy sells, benchmark sells proportional NIFTY
            strategy_total_investment = sum([pos['total_cost'] for pos in self.portfolio.values()])
            if strategy_total_investment > 0:
                # Calculate what proportion to sell
                nifty_position = self.benchmark_portfolio['NIFTY']
                sell_proportion = trade_amount / (strategy_total_investment + trade_amount)
                
                quantity_to_sell = nifty_position['quantity'] * sell_proportion
                sell_value = quantity_to_sell * nifty_price
                
                self.benchmark_cash += sell_value
                self.benchmark_portfolio['NIFTY']['quantity'] -= quantity_to_sell
                self.benchmark_portfolio['NIFTY']['total_cost'] -= (sell_value * nifty_position['avg_price'] / nifty_price)
                
                # Remove position if quantity becomes negligible
                if self.benchmark_portfolio['NIFTY']['quantity'] < 0.001:
                    del self.benchmark_portfolio['NIFTY']

    def backtest(self, start_date, end_date, initial_capital=1000000):
        """Run the complete backtest with fair benchmark"""
        # Get data
        data, successful_symbols = self.get_nifty50_data(start_date, end_date)
        nifty_data = self.get_benchmark_data(start_date, end_date)
        
        if not data:
            st.error("No data available for backtesting")
            return None, None
        
        # Calculate signals
        signals = self.calculate_signals(data)
        if not signals:
            st.error("No signals calculated - check data format")
            return None, None
        
        # Initialize
        self.cash = initial_capital
        self.benchmark_cash = initial_capital
        self.portfolio = {}
        self.benchmark_portfolio = {}
        self.trades = []
        self.equity_curve = []
        self.benchmark_equity_curve = []
        
        # Get all trading dates
        all_dates = set()
        for df in signals.values():
            all_dates.update(df.index)
        trading_dates = sorted(list(all_dates))
        
        if len(trading_dates) < 21:
            st.error("Not enough trading days for backtesting")
            return None, None
        
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
            symbols_to_remove = []
            for symbol, position in self.portfolio.items():
                if symbol in signals and date in signals[symbol].index: # and symbol not in self.core_etfs:
                    try:
                        current_price = signals[symbol].loc[date, 'Close']
                        profit_pct = (current_price - position['avg_price']) / position['avg_price']
                        ema49 = signals[symbol].loc[date, 'MA49']
                        
                        # idx = signals[symbol].index.get_loc(date)
                        # if idx > 0:
                        #     prev_ema49 = signals[symbol].iloc[idx - 1]['MA49']
                        #     prev_price = signals[symbol].iloc[idx - 1]['Close']
                        # else:
                        #     prev_ema49 = ema49
                        #     prev_price = current_price

                        #profit_last_buy = (current_price - position['last_buy_price']) / position['last_buy_price']
                        #rsi = signals[symbol].loc[date, 'RSI_14']   
                        
                        if (profit_pct >= self.target_percent and current_price < ema49) or (profit_pct <= -self.stop_loss_percent):
                            # Sell the position
                            sell_value = position['quantity'] * current_price
                            self.cash += sell_value
                            self.cash_deployed -= sell_value
                            
                            # Update benchmark (sell equivalent)
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
                            #self.buy_signal_stocks.remove(symbol)
                    except Exception as e:
                        continue
            
            # Remove sold positions
            for symbol in symbols_to_remove:
                del self.portfolio[symbol]
            
            # BUY LOGIC
            held_symbols = list(self.portfolio.keys())
            # weakest_stocks = self.find_weakest_stocks(signals, date, held_symbols)
            rsi_crossunder_35_stocks = self.find_rsi_crossunder_35_stocks(signals, date)
            
            # NEW BUY Signal
            new_buys = 0
            if rsi_crossunder_35_stocks:
                for symbol in rsi_crossunder_35_stocks:
                    if date in signals[symbol].index:
                        try:
                            if symbol not in self.buy_signal_stocks:
                                self.buy_signal_stocks.append(symbol)
                                # if symbol == 'AUTOBEES.NS':
                                #     print("New Buy Signal", date, symbol)
                                continue
                        except Exception as e:
                            continue
            
            #Actual buy
            averaging_stocks = {}
            for symbol in self.buy_signal_stocks:
                if date in signals[symbol].index:
                    buy_condition_matches = self.find_buy_condition_matches(signals, symbol, date)
                    # if symbol == 'AUTOBEES.NS':
                    #     print("Outside buy", date, symbol, buy_condition_matches)

                    if buy_condition_matches:
                        self.buy_signal_stocks.remove(symbol)
                        if symbol not in held_symbols and self.cash >= self.capital_per_trade:
                            price = signals[symbol].loc[date, 'Close']
                            quantity = int(self.capital_per_trade / price)
                            if quantity > 0:
                                cost = quantity * price
                                self.cash -= cost
                                self.cash_deployed += cost

                                # Update benchmark (buy equivalent NIFTY)
                                self.update_benchmark(date, cost, nifty_data, 'BUY')
                                
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
                            # if symbol == 'AUTOBEES.NS':
                            #     print("Averaging",date, symbol)
                            current_price = signals[symbol].loc[date, 'Close']
                            quantity = int(self.capital_per_trade / current_price)
                            old_position = self.portfolio[symbol]

                            if self.cash >= self.capital_per_trade and quantity > 0 and old_position['average_count'] < 25:
                                cost = quantity * current_price
                                self.cash -= cost
                                self.cash_deployed += cost
                                # Update benchmark (buy equivalent NIFTY)
                                self.update_benchmark(date, cost, nifty_data, 'BUY')
                                
                                # Update position with new average
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

            # Recalculate final portfolio values for equity curve
            final_portfolio_value = self.cash
            final_benchmark_value = self.benchmark_cash
            
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
            
            # Update equity curves
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
            
            progress_bar.progress((i + 1) / len(trading_dates))
        
        # Store signals for current positions display
        self.last_signals = signals
        
        return pd.DataFrame(self.equity_curve), pd.DataFrame(self.benchmark_equity_curve)
    
    def calculate_metrics(self, equity_curve, initial_capital, cash_deployed):
        """Calculate performance metrics"""
        if equity_curve.empty:
            return {}
        
        final_value = equity_curve['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate CAGR
        days = (equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days
        years = days / 365.25
        cagr = (final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # Win rate
        profitable_trades = [t for t in self.trades if t['action'] == 'SELL' and t['profit_amount'] > 0]
        total_sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        win_rate = len(profitable_trades) / len(total_sell_trades) if total_sell_trades else 0
        
        # Maximum drawdown
        equity_curve['rolling_max'] = equity_curve['portfolio_value'].expanding().max()
        equity_curve['drawdown'] = (equity_curve['portfolio_value'] - equity_curve['rolling_max']) / equity_curve['rolling_max']
        max_drawdown = equity_curve['drawdown'].min()
        
        return {
            'Total Return': f"{total_return:.2%}",
            'CAGR': f"{cagr:.2%}",
            'Final Value': f"₹{final_value:,.0f}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Trades': len(total_sell_trades),
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Current Positions': len(self.portfolio)
        }

def main():
    st.set_page_config(page_title="NIFTY SHOP Strategy Backtester - 7", layout="wide")
    
    st.title("🏪 NIFTY SHOP Strategy Backtester - 7 Buy RSI Below 35 + On reversal")
    st.markdown("**Based on NIFTY ETF Shop Strategy**")
    
    # # Add explanation of fair benchmark
    # with st.expander("ℹ️ About Fair Benchmark Comparison"):
    #     st.markdown("""
    #     **Fair Benchmark Methodology:**
    #     - **Traditional Approach**: Invests all capital in NIFTY 50 from day 1
    #     - **Fair Approach**: Only invests in NIFTY 50 when the strategy makes trades
    #     - When strategy BUYS ₹25,000 of ETF → Benchmark BUYS ₹25,000 of NIFTY 50
    #     - When strategy SELLS → Benchmark SELLS proportional NIFTY 50
    #     - This ensures both start with same cash and deploy capital at the same pace
    #     """)
    
    # Sidebar for parameters
    st.sidebar.header("Strategy Parameters")
    
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=1097))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=1200000, step=10000)
    capital_per_trade = st.sidebar.number_input("Capital per Trade (₹)", value=10000, step=1000)
    target_percent = st.sidebar.number_input("Target Profit (%)", value=9.00, step=0.1)
    stop_loss_percent = st.sidebar.number_input("Stop Loss (%)", value=1000.00, step=0.1)
    # averaging_threshold = st.sidebar.number_input("Averaging Threshold (%)", value=3.50, step=0.1)
    
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            strategy = NiftyShopStrategy(
                capital_per_trade=capital_per_trade,
                target_percent=target_percent,
                stop_loss_percent=stop_loss_percent,
                #averaging_threshold=averaging_threshold
            )
            
            equity_curve, benchmark_curve = strategy.backtest(start_date, end_date, initial_capital)
            
            if equity_curve is not None and not equity_curve.empty:
                # Calculate metrics for both strategy and benchmark
                strategy_metrics = strategy.calculate_metrics(equity_curve, initial_capital, strategy.cash_deployed)
                
                if not benchmark_curve.empty:
                    benchmark_final_value = benchmark_curve['portfolio_value'].iloc[-1]
                    benchmark_return = (benchmark_final_value - initial_capital) / initial_capital
                    
                    days = (benchmark_curve['date'].iloc[-1] - benchmark_curve['date'].iloc[0]).days
                    years = days / 365.25
                    benchmark_cagr = (benchmark_final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0
                else:
                    benchmark_return = 0
                    benchmark_cagr = 0
                    benchmark_final_value = initial_capital
                
                # Display metrics comparison
                st.header("📊 Performance Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🎯 Strategy Performance")
                    st.metric("Total Return", strategy_metrics['Total Return'])
                    st.metric("CAGR", strategy_metrics['CAGR'])
                    st.metric("Final Value", strategy_metrics['Final Value'])
                    st.metric("Win Rate", strategy_metrics['Win Rate'])
                    st.metric("Max Drawdown", strategy_metrics['Max Drawdown'])
                
                with col2:
                    st.subheader("📈 Fair Benchmark (NIFTY 50)")
                    st.metric("Total Return", f"{benchmark_return:.2%}")
                    st.metric("CAGR", f"{benchmark_cagr:.2%}")
                    st.metric("Final Value", f"₹{benchmark_final_value:,.0f}")
                    st.metric("Win Rate", "N/A")
                    st.metric("Max Drawdown", "N/A")
                
                with col3:
                    st.subheader("🏆 Strategy vs Benchmark")
                    excess_return = float(strategy_metrics['Total Return'].strip('%')) - (benchmark_return * 100)
                    excess_cagr = float(strategy_metrics['CAGR'].strip('%')) - (benchmark_cagr * 100)
                    
                    st.metric("Excess Return", f"{excess_return:.2f}%", 
                             f"{excess_return:.2f}%")
                    st.metric("Excess CAGR", f"{excess_cagr:.2f}%",
                             f"{excess_cagr:.2f}%")
                    st.metric("Total Trades", strategy_metrics['Total Trades'])
                    st.metric("Current Positions", strategy_metrics['Current Positions'])
                
                # Plot equity curves
                st.header("📈 Strategy vs Fair Benchmark Performance")
                
                fig = go.Figure()
                
                # Strategy equity curve
                fig.add_trace(go.Scatter(
                    x=equity_curve['date'],
                    y=equity_curve['portfolio_value'],
                    mode='lines',
                    name='NIFTY SHOP Strategy',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # Fair benchmark curve
                if not benchmark_curve.empty:
                    fig.add_trace(go.Scatter(
                        x=benchmark_curve['date'],
                        y=benchmark_curve['portfolio_value'],
                        mode='lines',
                        name='Fair NIFTY 50 Benchmark',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time - Strategy vs Fair Benchmark",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (₹)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show deployment comparison
                st.header("💰 Capital Deployment Comparison")
                
                fig2 = go.Figure()
                
                # Strategy cash levels
                fig2.add_trace(go.Scatter(
                    x=equity_curve['date'],
                    y=equity_curve['cash'],
                    mode='lines',
                    name='Strategy Cash',
                    line=dict(color='#2ca02c', width=2)
                ))
                
                # Benchmark cash levels
                if not benchmark_curve.empty:
                    fig2.add_trace(go.Scatter(
                        x=benchmark_curve['date'],
                        y=benchmark_curve['cash'],
                        mode='lines',
                        name='Benchmark Cash',
                        line=dict(color='#d62728', width=2)
                    ))
                
                fig2.update_layout(
                    title="Cash Levels Over Time - Showing Fair Capital Deployment",
                    xaxis_title="Date",
                    yaxis_title="Cash (₹)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Trade history
                st.header("📋 Recent Trades")
                if strategy.trades:
                    trades_df = pd.DataFrame(strategy.trades)
                    trades_df = trades_df.sort_values('date', ascending=False)
                    
                    # Format for display
                    display_df = trades_df.copy()
                    display_df['value'] = display_df['value'].apply(lambda x: f"₹{x:,.0f}")
                    display_df['price'] = display_df['price'].apply(lambda x: f"₹{x:.2f}")
                    display_df['profit_pct'] = display_df['profit_pct'].apply(lambda x: f"{x:.2f}%")
                    display_df['profit_amount'] = display_df['profit_amount'].apply(lambda x: f"₹{x:,.0f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No trades executed during the backtest period")
                
                # Current positions
                st.header("💼 Current Positions")
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
                        st.dataframe(positions_df, use_container_width=True)
                        
                        # Show summary
                        total_investment = sum([pos['total_cost'] for pos in strategy.portfolio.values()])
                        st.info(f"**Total Positions:** {len(positions_data)} | **Total Investment:** ₹{total_investment:,.0f}")
                    else:
                        st.warning("Could not display position details due to data issues")
                else:
                    st.info("No current positions")

                # Buy Signal Symbol
                st.header("Buy Signal Symbol")
                if strategy.buy_signal_stocks:
                    #buy_signals = [symbol for symbol, signal in strategy.buy_signal_stocks.items() if signal == 'BUY']
                    st.write(strategy.buy_signal_stocks)
                else:
                    st.info("No buy signals")

            else:
                st.error("Backtesting failed. Please check your parameters and try again.")

if __name__ == "__main__":
    main()
