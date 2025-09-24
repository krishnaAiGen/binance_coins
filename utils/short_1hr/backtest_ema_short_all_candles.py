#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EMAShortAllCandlesBacktester:
    def __init__(self, data_folder="/Users/krishnayadav/Documents/test_projects/binance_coins/utils/short_1hr/data", initial_capital=2000.0):
        self.data_folder = data_folder
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.coin_results = {}
        self.all_trades = []
        
        # Load available symbols from data folder
        self.symbols = self._load_symbols()
        
    def _load_symbols(self):
        """Load symbols from CSV files in the data folder and sort by launch date"""
        if not os.path.exists(self.data_folder):
            print(f"❌ Data folder '{self.data_folder}' not found!")
            return []
        
        csv_files = [f for f in os.listdir(self.data_folder) 
                    if f.endswith('.csv') and '_1h_7d_' in f and not f.endswith('_backup.csv')]
        symbol_data = []
        
        for filename in csv_files:
            try:
                # Extract symbol and launch date from filename like "BTCUSDT_1h_7d_2017-08-17_10-00-00.csv"
                parts = filename.split('_1h_7d_')
                if len(parts) == 2:
                    symbol = parts[0]
                    date_part = parts[1].replace('.csv', '').replace('-', ':')
                    
                    # Parse launch date from filename
                    try:
                        # Convert "2017-08-17_10:00:00" to datetime
                        date_str = date_part.replace('_', ' ')
                        launch_date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        symbol_data.append((symbol, launch_date, filename))
                    except:
                        # Fallback: try to get date from the actual CSV file
                        filepath = os.path.join(self.data_folder, filename)
                        df = pd.read_csv(filepath)
                        if 'launch_date' in df.columns:
                            launch_date_str = df['launch_date'].iloc[0]
                            launch_date = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
                            symbol_data.append((symbol, launch_date, filename))
                        else:
                            # If we can't parse date, use a very late date so it goes last
                            launch_date = datetime(2099, 1, 1)
                            symbol_data.append((symbol, launch_date, filename))
            except Exception as e:
                print(f"⚠️  Warning: Could not parse date for {filename}: {e}")
                # Add with a very late date so it goes last
                symbol = filename.split('_1h_7d_')[0] if '_1h_7d_' in filename else filename
                launch_date = datetime(2099, 1, 1)
                symbol_data.append((symbol, launch_date, filename))
        
        # Sort by launch date (earliest first)
        symbol_data.sort(key=lambda x: x[1])
        
        # Extract just the symbols in chronological order
        symbols = [item[0] for item in symbol_data]
        
        print(f"📊 Found {len(symbols)} symbols with 1-hour 7-day data")
        print(f"📅 Sorted chronologically by launch date")
        
        if symbol_data:
            earliest = symbol_data[0]
            latest = symbol_data[-1]
            print(f"   Earliest: {earliest[0]} ({earliest[1].strftime('%Y-%m-%d %H:%M')})")
            print(f"   Latest:   {latest[0]} ({latest[1].strftime('%Y-%m-%d %H:%M')})")
            
            # Check for future dates (2025+) and warn user
            future_dates = [item for item in symbol_data if item[1].year >= 2025]
            if future_dates:
                print(f"⚠️  Warning: {len(future_dates)} coins have future launch dates (2025+)")
                print(f"   This may indicate test/simulated data")
        
        return symbols
    
    def load_symbol_data(self, symbol):
        """Load 1-hour data for a specific symbol"""
        csv_files = [f for f in os.listdir(self.data_folder) 
                    if f.startswith(f"{symbol}_1h_7d_") and f.endswith('.csv') and not f.endswith('_backup.csv')]
        
        if not csv_files:
            print(f"❌ No data found for {symbol}")
            return None
        
        filename = os.path.join(self.data_folder, csv_files[0])
        
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Check if EMA columns exist
            required_emas = ['ema_5', 'ema_10', 'ema_20', 'ema_50']
            missing_emas = [ema for ema in required_emas if ema not in df.columns]
            
            if missing_emas:
                print(f"❌ {symbol}: Missing EMA columns: {missing_emas}")
                print("💡 Please run add_ema.py first to calculate EMAs")
                return None
            
            return df
        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")
            return None
    
    def calculate_ema(self, data, period):
        """Calculate EMA for given period"""
        return data.ewm(span=period, adjust=False).mean()
    
    def simulate_ema_short_trade_all_candles(self, df, symbol, ema_period=5, target_pct=15.0, stop_loss_pct=5.0, start_scan_hour=1):
        """
        Simulate EMA short trading strategy - scan ALL candles for signal:
        - Start scanning from specified hour (default: hour 1)
        - Check each candle: if close < EMA, enter SHORT at next candle
        - Stop scanning once signal found and trade executed
        - If no signal found by end of 7 days, return "No Signal"
        - TP: target_pct% gain (price goes down)
        - SL: stop_loss_pct% loss (price goes up)
        """
        # Check if we have enough candles
        if len(df) < start_scan_hour + 1:
            return None
        
        # Check if the specified EMA column exists, if not calculate it
        ema_col = f'ema_{ema_period}'
        if ema_col not in df.columns:
            df[ema_col] = self.calculate_ema(df['close'], ema_period)
        
        # Get launch date from data
        launch_date = df.iloc[0]['launch_date'] if 'launch_date' in df.columns else 'Unknown'
        
        # Scan all candles starting from start_scan_hour
        for scan_hour in range(start_scan_hour, len(df)):
            signal_index = scan_hour
            entry_index = scan_hour + 1
            
            # Check if we have data for entry
            if entry_index >= len(df):
                break  # No more candles for entry
            
            # Signal detection: Check if current candle closed below EMA
            signal_candle = df.iloc[signal_index]
            ema_value = signal_candle[ema_col]
            
            # Skip if EMA is NaN
            if pd.isna(ema_value):
                continue
            
            # Check if signal is valid (close below EMA for SHORT)
            if signal_candle['close'] < ema_value:
                # 🚨 SIGNAL FOUND! Execute trade
                
                # Entry candle: Enter at next hour after signal
                entry_candle = df.iloc[entry_index]
                entry_price = entry_candle['open']
                entry_time = entry_candle['timestamp']
                
                # Calculate target and stop loss prices for SHORT position
                target_price = entry_price * (1 - target_pct / 100)    # Target: price goes DOWN
                stop_loss_price = entry_price * (1 + stop_loss_pct / 100)  # Stop: price goes UP
                
                # Calculate position size based on current capital
                position_size = self.current_capital / entry_price
                
                # Monitor trade from entry candle onwards
                for i in range(entry_index, len(df)):
                    current_candle = df.iloc[i]
                    
                    # For the entry candle, check if SL hits first (more conservative)
                    if i == entry_index:
                        # Check entry candle: prioritize stop loss if both TP and SL would hit
                        if current_candle['high'] >= stop_loss_price:
                            exit_price = stop_loss_price
                            exit_time = current_candle['timestamp']
                            profit_usd = position_size * (entry_price - exit_price)  # Will be negative (loss)
                            profit_pct = ((entry_price - exit_price) / entry_price) * 100
                            
                            trade = {
                                'symbol': symbol,
                                'launch_date': launch_date,
                                'entry_signal': True,
                                'signal_found_at_hour': scan_hour,
                                'ema_period': ema_period,
                                'ema_value_at_signal': ema_value,
                                'close_at_signal': signal_candle['close'],
                                'entry_time': entry_time,
                                'exit_time': exit_time,
                                'entry_price': entry_price,
                                'target_price': target_price,
                                'stop_loss_price': stop_loss_price,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'profit_usd': profit_usd,
                                'profit_pct': profit_pct,
                                'exit_reason': 'Stop Loss (Same Candle)',
                                'capital_before': self.current_capital,
                                'capital_after': self.current_capital + profit_usd,
                                'duration_hours': 0,
                                'signal_reason': f"Close {signal_candle['close']:.6f} < EMA{ema_period} {ema_value:.6f}"
                            }
                            
                            # Update capital
                            self.current_capital += profit_usd
                            return trade
                            
                        elif current_candle['low'] <= target_price:
                            exit_price = target_price
                            exit_time = current_candle['timestamp']
                            profit_usd = position_size * (entry_price - exit_price)  # Will be positive (profit)
                            profit_pct = ((entry_price - exit_price) / entry_price) * 100
                            
                            trade = {
                                'symbol': symbol,
                                'launch_date': launch_date,
                                'entry_signal': True,
                                'signal_found_at_hour': scan_hour,
                                'ema_period': ema_period,
                                'ema_value_at_signal': ema_value,
                                'close_at_signal': signal_candle['close'],
                                'entry_time': entry_time,
                                'exit_time': exit_time,
                                'entry_price': entry_price,
                                'target_price': target_price,
                                'stop_loss_price': stop_loss_price,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'profit_usd': profit_usd,
                                'profit_pct': profit_pct,
                                'exit_reason': 'Target Hit (Same Candle)',
                                'capital_before': self.current_capital,
                                'capital_after': self.current_capital + profit_usd,
                                'duration_hours': 0,
                                'signal_reason': f"Close {signal_candle['close']:.6f} < EMA{ema_period} {ema_value:.6f}"
                            }
                            
                            # Update capital
                            self.current_capital += profit_usd
                            return trade
                            
                        # If neither TP nor SL hit on entry candle, continue to next candle
                        continue
                    
                    # For subsequent candles
                    # Check if target hit (price went DOWN - low <= target_price)
                    if current_candle['low'] <= target_price:
                        exit_price = target_price
                        exit_time = current_candle['timestamp']
                        profit_usd = position_size * (entry_price - exit_price)  # Positive profit
                        profit_pct = ((entry_price - exit_price) / entry_price) * 100
                        
                        trade = {
                            'symbol': symbol,
                            'launch_date': launch_date,
                            'entry_signal': True,
                            'signal_found_at_hour': scan_hour,
                            'ema_period': ema_period,
                            'ema_value_at_signal': ema_value,
                            'close_at_signal': signal_candle['close'],
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'target_price': target_price,
                            'stop_loss_price': stop_loss_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'profit_usd': profit_usd,
                            'profit_pct': profit_pct,
                            'exit_reason': 'Target Hit',
                            'capital_before': self.current_capital,
                            'capital_after': self.current_capital + profit_usd,
                            'duration_hours': i - entry_index,
                            'signal_reason': f"Close {signal_candle['close']:.6f} < EMA{ema_period} {ema_value:.6f}"
                        }
                        
                        # Update capital
                        self.current_capital += profit_usd
                        return trade
                    
                    # Check if stop loss hit (price went UP - high >= stop_loss_price)
                    elif current_candle['high'] >= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = current_candle['timestamp']
                        profit_usd = position_size * (entry_price - exit_price)  # Negative profit (loss)
                        profit_pct = ((entry_price - exit_price) / entry_price) * 100
                        
                        trade = {
                            'symbol': symbol,
                            'launch_date': launch_date,
                            'entry_signal': True,
                            'signal_found_at_hour': scan_hour,
                            'ema_period': ema_period,
                            'ema_value_at_signal': ema_value,
                            'close_at_signal': signal_candle['close'],
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'target_price': target_price,
                            'stop_loss_price': stop_loss_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'profit_usd': profit_usd,
                            'profit_pct': profit_pct,
                            'exit_reason': 'Stop Loss',
                            'capital_before': self.current_capital,
                            'capital_after': self.current_capital + profit_usd,
                            'duration_hours': i - entry_index,
                            'signal_reason': f"Close {signal_candle['close']:.6f} < EMA{ema_period} {ema_value:.6f}"
                        }
                        
                        # Update capital
                        self.current_capital += profit_usd
                        return trade
                
                # If neither target nor stop loss hit, exit at last price (end of 7d period)
                last_candle = df.iloc[-1]
                exit_price = last_candle['close']
                exit_time = last_candle['timestamp']
                profit_usd = position_size * (entry_price - exit_price)
                profit_pct = ((entry_price - exit_price) / entry_price) * 100
                
                trade = {
                    'symbol': symbol,
                    'launch_date': launch_date,
                    'entry_signal': True,
                    'signal_found_at_hour': scan_hour,
                    'ema_period': ema_period,
                    'ema_value_at_signal': ema_value,
                    'close_at_signal': signal_candle['close'],
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'stop_loss_price': stop_loss_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'profit_usd': profit_usd,
                    'profit_pct': profit_pct,
                    'exit_reason': '7d Timeout',
                    'capital_before': self.current_capital,
                    'capital_after': self.current_capital + profit_usd,
                    'duration_hours': len(df) - 1 - entry_index,
                    'signal_reason': f"Close {signal_candle['close']:.6f} < EMA{ema_period} {ema_value:.6f}"
                }
                
                # Update capital
                self.current_capital += profit_usd
                return trade
        
        # No signal found in entire 7-day period
        return {
            'symbol': symbol,
            'launch_date': launch_date,
            'entry_signal': False,
            'signal_found_at_hour': None,
            'signal_reason': f"No close < EMA{ema_period} signal found in {len(df)} hours",
            'exit_reason': 'No Signal',
            'profit_usd': 0,
            'profit_pct': 0,
            'hours_scanned': len(df) - start_scan_hour
        }
    
    def backtest_all_symbols(self, ema_period=5, target_pct=15.0, stop_loss_pct=5.0, start_scan_hour=1):
        """Run backtest on all available symbols with EMA short strategy - scan all candles"""
        print(f"🚀 Starting EMA Short backtest (ALL CANDLES) on {len(self.symbols)} symbols")
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"📉 Strategy: SHORT when close < EMA-{ema_period} (scan all hours)")
        print(f"🎯 Target: {target_pct}% | 🛑 Stop Loss: {stop_loss_pct}%")
        print(f"📊 Start scanning from: Hour #{start_scan_hour}")
        print("=" * 80)
        
        target_hits = 0
        stop_losses = 0
        timeouts = 0
        no_signals = 0
        successful_symbols = 0
        failed_symbols = 0
        total_hours_scanned = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            # Load data for symbol first to get launch date
            df = self.load_symbol_data(symbol)
            
            if df is None or len(df) < 2:
                print(f"[{i:3d}/{len(self.symbols):3d}] {symbol:<12} ❌ Insufficient data")
                failed_symbols += 1
                continue
            
            # Get launch date for display
            launch_date_str = df.iloc[0]['launch_date'] if 'launch_date' in df.columns else 'Unknown'
            launch_date_short = launch_date_str[:10] if launch_date_str != 'Unknown' else 'Unknown'
            
            print(f"[{i:3d}/{len(self.symbols):3d}] {symbol:<12} ({launch_date_short}) ", end="")
            
            # Run trade simulation
            trade = self.simulate_ema_short_trade_all_candles(df, symbol, ema_period, target_pct, stop_loss_pct, start_scan_hour)
            
            if trade:
                self.all_trades.append(trade)
                
                # Count exit reasons
                if not trade.get('entry_signal', False):
                    no_signals += 1
                    hours_scanned = trade.get('hours_scanned', 0)
                    total_hours_scanned += hours_scanned
                    print(f"🚫 No Signal (scanned {hours_scanned}h)")
                elif trade['exit_reason'] == 'Target Hit' or 'Target Hit' in trade['exit_reason']:
                    target_hits += 1
                    signal_hour = trade.get('signal_found_at_hour', '?')
                    print(f"🎯 TP: ${trade['profit_usd']:+7.0f} (signal@h{signal_hour}) | Capital: ${self.current_capital:,.0f}")
                elif trade['exit_reason'] == 'Stop Loss' or 'Stop Loss' in trade['exit_reason']:
                    stop_losses += 1
                    signal_hour = trade.get('signal_found_at_hour', '?')
                    print(f"🛑 SL: ${trade['profit_usd']:+7.0f} (signal@h{signal_hour}) | Capital: ${self.current_capital:,.0f}")
                else:
                    timeouts += 1
                    signal_hour = trade.get('signal_found_at_hour', '?')
                    print(f"⏰ TO: ${trade['profit_usd']:+7.0f} (signal@h{signal_hour}) | Capital: ${self.current_capital:,.0f}")
                
                successful_symbols += 1
            else:
                print("❌ Trade simulation failed")
                failed_symbols += 1
        
        # Final statistics
        print("\n" + "=" * 80)
        print("📊 EMA SHORT BACKTEST RESULTS SUMMARY (ALL CANDLES)")
        print("=" * 80)
        
        total_trades = len(self.all_trades)
        trading_trades = len([t for t in self.all_trades if t.get('entry_signal', False)])
        total_profit = self.current_capital - self.initial_capital
        total_return = (total_profit / self.initial_capital) * 100
        
        print(f"💰 Portfolio Performance:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Capital: ${self.current_capital:,.2f}")
        print(f"   Total Profit/Loss: ${total_profit:+,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        
        print(f"\n📈 Trade Analysis:")
        print(f"   Total Symbols: {total_trades}")
        print(f"   Trading Signals: {trading_trades}")
        print(f"   No Signals: {no_signals}")
        print(f"   Failed Symbols: {failed_symbols}")
        print(f"   Signal Rate: {trading_trades/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
        
        if no_signals > 0:
            avg_scan_hours = total_hours_scanned / no_signals
            print(f"   Avg Hours Scanned (no signal): {avg_scan_hours:.1f}")
        
        if trading_trades > 0:
            print(f"\n🎯 Exit Reasons (Trading only):")
            print(f"   Target Hits: {target_hits} ({target_hits/trading_trades*100:.1f}%)")
            print(f"   Stop Losses: {stop_losses} ({stop_losses/trading_trades*100:.1f}%)")
            print(f"   Timeouts: {timeouts} ({timeouts/trading_trades*100:.1f}%)")
            
            # Calculate win rate (target hits / trading trades)
            win_rate = (target_hits / trading_trades) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
            
            # Average profit per trade
            trading_profit = sum([t['profit_usd'] for t in self.all_trades if t.get('entry_signal', False)])
            avg_profit = trading_profit / trading_trades
            print(f"   Average Profit/Trade: ${avg_profit:+.2f}")
            
            # Signal timing analysis
            signal_hours = [t.get('signal_found_at_hour', 0) for t in self.all_trades if t.get('entry_signal', False)]
            if signal_hours:
                avg_signal_hour = sum(signal_hours) / len(signal_hours)
                print(f"   Average Signal Hour: {avg_signal_hour:.1f}")
                print(f"   Earliest Signal: Hour {min(signal_hours)}")
                print(f"   Latest Signal: Hour {max(signal_hours)}")
        
        return {
            'total_symbols': total_trades,
            'trading_signals': trading_trades,
            'no_signals': no_signals,
            'target_hits': target_hits,
            'stop_losses': stop_losses,
            'timeouts': timeouts,
            'win_rate': (target_hits / trading_trades) * 100 if trading_trades > 0 else 0,
            'signal_rate': (trading_trades / total_trades) * 100 if total_trades > 0 else 0,
            'total_profit': total_profit,
            'total_return': total_return,
            'final_capital': self.current_capital
        }
    
    def save_detailed_results(self, filename_suffix=""):
        """Save detailed trade results to CSV"""
        if not self.all_trades:
            print("No trades to save")
            return
        
        df = pd.DataFrame(self.all_trades)
        
        # Format datetime columns
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Round numerical columns
        numerical_cols = ['ema_value_at_signal', 'close_at_signal', 'entry_price', 'target_price', 'stop_loss_price', 
                         'exit_price', 'position_size', 'profit_usd', 'profit_pct', 'capital_before', 'capital_after']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ema_short_all_candles_backtest{filename_suffix}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        # Show sample trades
        trading_trades = df[df.get('entry_signal', True) == True]
        if len(trading_trades) > 0:
            print(f"\n📋 Sample Trades (First 5 with signals):")
            sample_cols = ['symbol', 'signal_found_at_hour', 'ema_period', 'close_at_signal', 'ema_value_at_signal', 'entry_price', 'exit_price', 'profit_usd', 'exit_reason', 'capital_after']
            sample_df = trading_trades[sample_cols].head().copy()
            
            # Round columns for better display
            for col in ['close_at_signal', 'ema_value_at_signal', 'entry_price', 'exit_price']:
                if col in sample_df.columns:
                    sample_df[col] = sample_df[col].round(6)
            
            print(sample_df.to_string(index=False))
        
        return filename

def main():
    print("🚀 === EMA Short Strategy Backtester (ALL CANDLES) ===")
    print("Strategy: SHORT when price closes below EMA - scan ALL hours!")
    print("Capital: $2000 compounded")
    print("Data: 1-hour intervals, 7-day periods from launch")
    print("=" * 80)
    
    # ===== CONFIGURABLE PARAMETERS =====
    DEFAULT_EMA = 5          # Default EMA period
    DEFAULT_TP = 45.0        # Default Target Profit %
    DEFAULT_SL = 10.0        # Default Stop Loss %
    DEFAULT_START_SCAN = 1   # Default hour to start scanning (1=second hour from launch)
    LAST_COINS = 200          # Number of coins to test (20, 50, 100, 200, etc.)
    # ====================================
    
    # Initialize backtester
    backtester = EMAShortAllCandlesBacktester(initial_capital=2000.0)
    
    if not backtester.symbols:
        print("❌ No 1-hour data found! Please run price_downloader.py first.")
        print("💡 Also ensure EMA indicators are calculated by running add_ema.py")
        return
    
    # Limit to LAST_COINS from chronologically sorted list (most recently launched)
    if LAST_COINS and LAST_COINS < len(backtester.symbols):
        original_count = len(backtester.symbols)
        backtester.symbols = backtester.symbols[-LAST_COINS:]  # Take last N coins
        print(f"🎯 Limited to last {LAST_COINS} coins (out of {original_count} available) - most recently launched")
    else:
        print(f"🎯 Using all {len(backtester.symbols)} available coins")
    
    while True:
        print("\nChoose an option:")
        print(f"1. 🎯 Run single backtest (EMA={DEFAULT_EMA}, TP={DEFAULT_TP}%, SL={DEFAULT_SL}%, Scan from Hour={DEFAULT_START_SCAN}, Coins={len(backtester.symbols)})")
        print("2. ⚙️  Run custom single backtest (specify your own EMA/TP/SL)")
        print("3. 🚪 Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            print(f"\n🚀 Running single EMA short backtest (ALL CANDLES)...")
            print(f"📈 EMA={DEFAULT_EMA}, TP={DEFAULT_TP}%, SL={DEFAULT_SL}%, Start Scan Hour={DEFAULT_START_SCAN}")
            
            # Run single backtest with default parameters
            results = backtester.backtest_all_symbols(ema_period=DEFAULT_EMA, target_pct=DEFAULT_TP, 
                                                     stop_loss_pct=DEFAULT_SL, start_scan_hour=DEFAULT_START_SCAN)
            
            # Save detailed results
            backtester.save_detailed_results()
            
            print(f"\n✅ EMA Short backtest (ALL CANDLES) complete!")
            print(f"🎯 Final Result: ${results['total_profit']:+,.2f} ({results['total_return']:+.2f}%)")
            print(f"📡 Signal Rate: {results['signal_rate']:.1f}%")
            print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
            
        elif choice == '2':
            try:
                custom_ema = int(input(f"Enter EMA Period (current default: {DEFAULT_EMA}): "))
                custom_tp = float(input(f"Enter Target Profit % (current default: {DEFAULT_TP}%): "))
                custom_sl = float(input(f"Enter Stop Loss % (current default: {DEFAULT_SL}%): "))
                custom_scan = int(input(f"Enter Start Scan Hour # (current default: {DEFAULT_START_SCAN}): "))
                
                print(f"\n🚀 Running custom EMA short backtest (ALL CANDLES)...")
                print(f"📈 EMA={custom_ema}, TP={custom_tp}%, SL={custom_sl}%, Start Scan Hour={custom_scan}")
                
                # Reset backtester state
                backtester.current_capital = backtester.initial_capital
                backtester.all_trades = []
                
                # Run single backtest with custom parameters
                results = backtester.backtest_all_symbols(ema_period=custom_ema, target_pct=custom_tp, 
                                                         stop_loss_pct=custom_sl, start_scan_hour=custom_scan)
                
                # Save detailed results
                backtester.save_detailed_results(filename_suffix=f"_EMA{custom_ema}_TP{custom_tp}_SL{custom_sl}")
                
                print(f"\n✅ Custom EMA short backtest (ALL CANDLES) complete!")
                print(f"🎯 Final Result: ${results['total_profit']:+,.2f} ({results['total_return']:+.2f}%)")
                print(f"📡 Signal Rate: {results['signal_rate']:.1f}%")
                print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
                
            except ValueError:
                print("❌ Invalid input. Please enter numeric values.")
            
        elif choice == '3':
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
