#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# OPTIMIZATION PARAMETERS
# ============================================================

# Target Profit percentages to test
TARGET_PROFITS = [5, 10, 15, 20, 25, 30, 35, 40]

# Stop Loss percentages to test  
STOP_LOSSES = [3, 5, 7, 10, 12, 15, 20, 25]

# Default parameters
DEFAULT_TARGET = 20  # 20%
DEFAULT_STOP_LOSS = 7  # 7%

class CryptoHourlyBacktester:
    def __init__(self, data_folder="utils/1_hr/1_hr"):
        self.data_folder = data_folder
        self.futures_list = self._load_futures_list()
        self.results = []
        
    def _load_futures_list(self):
        """Load the list of futures contracts"""
        with open('utils/futures.json', 'r') as f:
            all_futures = json.load(f)
        return all_futures[-200:]  # Last 200 coins
    
    def load_symbol_data(self, symbol):
        """Load price data for a specific symbol"""
        filename = f"{self.data_folder}/{symbol}_1h.csv"
        
        if not os.path.exists(filename):
            print(f"Warning: Data file not found for {symbol}")
            return None
        
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Ensure we have required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {symbol}")
                return None
            
            # Calculate green/red candles
            df['is_green'] = df['close'] > df['open']
            df['is_red'] = df['close'] < df['open']
            df['candle_type'] = df.apply(lambda row: 'green' if row['is_green'] else 'red' if row['is_red'] else 'doji', axis=1)
            
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None
    
    def simulate_trade(self, df, entry_idx, trade_type, target_pct, stop_loss_pct):
        """
        Simulate a single trade
        trade_type: 'long' for green candle, 'short' for red candle
        target_pct: target profit percentage
        stop_loss_pct: stop loss percentage
        """
        if entry_idx >= len(df) - 1:
            return None
        
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row['close']  # Enter at close of signal candle
        entry_time = entry_row['timestamp']
        
        if trade_type == 'long':
            # LONG position: profit when price goes up
            target_price = entry_price * (1 + target_pct / 100)
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        else:  # 'short'
            # SHORT position: profit when price goes down
            target_price = entry_price * (1 - target_pct / 100)
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
        
        # Look for exit in subsequent candles
        for i in range(entry_idx + 1, len(df)):
            current_row = df.iloc[i]
            current_time = current_row['timestamp']
            
            if trade_type == 'long':
                # Check target hit (price went up)
                if current_row['high'] >= target_price:
                    profit_pct = ((target_price - entry_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_pct,  # Assuming $100 position
                        'exit_reason': 'Target Hit',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
                
                # Check stop loss hit (price went down)
                if current_row['low'] <= stop_loss_price:
                    profit_pct = ((stop_loss_price - entry_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_pct,
                        'exit_reason': 'Stop Loss',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
            
            else:  # 'short'
                # Check target hit (price went down)
                if current_row['low'] <= target_price:
                    profit_pct = ((entry_price - target_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_pct,
                        'exit_reason': 'Target Hit',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
                
                # Check stop loss hit (price went up)
                if current_row['high'] >= stop_loss_price:
                    profit_pct = ((entry_price - stop_loss_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_pct,
                        'exit_reason': 'Stop Loss',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
        
        # If no exit condition met, exit at last available price
        last_row = df.iloc[-1]
        if trade_type == 'long':
            profit_pct = ((last_row['close'] - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - last_row['close']) / entry_price) * 100
        
        return {
            'entry_time': entry_time,
            'exit_time': last_row['timestamp'],
            'entry_price': entry_price,
            'exit_price': last_row['close'],
            'trade_type': trade_type,
            'profit_pct': profit_pct,
            'profit_usd': profit_pct,
            'exit_reason': 'End of Data',
            'duration_hours': (last_row['timestamp'] - entry_time).total_seconds() / 3600,
            'target_pct': target_pct,
            'stop_loss_pct': stop_loss_pct
        }
    
    def backtest_symbol(self, symbol, target_pct=DEFAULT_TARGET, stop_loss_pct=DEFAULT_STOP_LOSS, start_date=None, end_date=None):
        """Backtest a single symbol with green/red candle strategy"""
        df = self.load_symbol_data(symbol)
        if df is None or len(df) < 2:
            return []
        
        # Filter by date range if specified
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        if len(df) < 2:
            return []
        
        trades = []
        
        # Look for green and red candles (skip doji candles)
        for i in range(len(df) - 1):
            current_candle = df.iloc[i]
            
            if current_candle['candle_type'] == 'green':
                # Green candle -> go LONG
                trade = self.simulate_trade(df, i, 'long', target_pct, stop_loss_pct)
                if trade:
                    trade['symbol'] = symbol
                    trades.append(trade)
            
            elif current_candle['candle_type'] == 'red':
                # Red candle -> go SHORT
                trade = self.simulate_trade(df, i, 'short', target_pct, stop_loss_pct)
                if trade:
                    trade['symbol'] = symbol
                    trades.append(trade)
        
        return trades
    
    def backtest_all_symbols(self, target_pct=DEFAULT_TARGET, stop_loss_pct=DEFAULT_STOP_LOSS, start_date=None, end_date=None):
        """Backtest all symbols"""
        print(f"Backtesting {len(self.futures_list)} symbols...")
        print(f"Strategy: Green candle = LONG, Red candle = SHORT")
        print(f"Target: {target_pct}%, Stop Loss: {stop_loss_pct}%")
        if start_date or end_date:
            print(f"Date range: {start_date or 'start'} to {end_date or 'end'}")
        print("=" * 60)
        
        all_trades = []
        successful_symbols = []
        failed_symbols = []
        
        for i, symbol in enumerate(self.futures_list, 1):
            print(f"[{i}/{len(self.futures_list)}] {symbol}...", end=" ")
            
            try:
                trades = self.backtest_symbol(symbol, target_pct, stop_loss_pct, start_date, end_date)
                if trades:
                    all_trades.extend(trades)
                    successful_symbols.append(symbol)
                    print(f"✓ {len(trades)} trades")
                else:
                    failed_symbols.append(symbol)
                    print("✗ No trades")
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"✗ Error: {e}")
        
        print("\n" + "=" * 60)
        print(f"Processed: {len(successful_symbols)} successful, {len(failed_symbols)} failed")
        print(f"Total trades: {len(all_trades)}")
        
        return all_trades
    
    def analyze_results(self, trades):
        """Analyze backtest results"""
        if not trades:
            print("No trades to analyze")
            return {}
        
        df = pd.DataFrame(trades)
        
        # Overall statistics
        total_trades = len(df)
        winning_trades = len(df[df['profit_pct'] > 0])
        losing_trades = len(df[df['profit_pct'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = df['profit_pct'].sum()
        avg_profit = df['profit_pct'].mean()
        
        # By trade type
        long_trades = df[df['trade_type'] == 'long']
        short_trades = df[df['trade_type'] == 'short']
        
        # By exit reason
        target_hits = len(df[df['exit_reason'] == 'Target Hit'])
        stop_losses = len(df[df['exit_reason'] == 'Stop Loss'])
        
        # Duration stats
        avg_duration = df['duration_hours'].mean()
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_profit': long_trades['profit_pct'].sum() if len(long_trades) > 0 else 0,
            'short_profit': short_trades['profit_pct'].sum() if len(short_trades) > 0 else 0,
            'target_hits': target_hits,
            'stop_losses': stop_losses,
            'avg_duration': avg_duration,
            'target_pct': df['target_pct'].iloc[0] if len(df) > 0 else 0,
            'stop_loss_pct': df['stop_loss_pct'].iloc[0] if len(df) > 0 else 0
        }
        
        return stats
    
    def print_results(self, trades, stats):
        """Print detailed results"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"📊 TRADE STATISTICS:")
        print(f"   Total trades: {stats['total_trades']}")
        print(f"   Winning trades: {stats['winning_trades']}")
        print(f"   Losing trades: {stats['losing_trades']}")
        print(f"   Win rate: {stats['win_rate']:.1f}%")
        
        print(f"\n💰 PROFIT & LOSS:")
        print(f"   Total profit: {stats['total_profit']:.2f}%")
        print(f"   Average profit per trade: {stats['avg_profit']:.2f}%")
        
        print(f"\n📈 TRADE BREAKDOWN:")
        print(f"   LONG trades: {stats['long_trades']} (profit: {stats['long_profit']:.2f}%)")
        print(f"   SHORT trades: {stats['short_trades']} (profit: {stats['short_profit']:.2f}%)")
        
        print(f"\n🎯 EXIT ANALYSIS:")
        print(f"   Target hits: {stats['target_hits']}")
        print(f"   Stop losses: {stats['stop_losses']}")
        print(f"   Average duration: {stats['avg_duration']:.1f} hours")
        
        print(f"\n⚙️  PARAMETERS:")
        print(f"   Target: {stats['target_pct']}%")
        print(f"   Stop Loss: {stats['stop_loss_pct']}%")
        
        # Show best performing symbols
        if trades:
            df = pd.DataFrame(trades)
            symbol_performance = df.groupby('symbol')['profit_pct'].agg(['count', 'sum', 'mean']).round(2)
            symbol_performance.columns = ['trades', 'total_profit', 'avg_profit']
            symbol_performance = symbol_performance.sort_values('total_profit', ascending=False)
            
            print(f"\n🏆 TOP 10 PERFORMING SYMBOLS:")
            print(symbol_performance.head(10).to_string())
            
            print(f"\n📉 WORST 5 PERFORMING SYMBOLS:")
            print(symbol_performance.tail(5).to_string())
    
    def optimize_parameters(self, target_profits=TARGET_PROFITS, stop_losses=STOP_LOSSES, max_symbols=20):
        """
        Optimize TP and SL parameters
        Test different combinations on a subset of symbols for speed
        """
        print(f"PARAMETER OPTIMIZATION")
        print(f"Testing {len(target_profits)} target profits × {len(stop_losses)} stop losses = {len(target_profits) * len(stop_losses)} combinations")
        print(f"Using first {max_symbols} symbols for speed")
        print("=" * 60)
        
        # Use subset of symbols for optimization
        test_symbols = self.futures_list[:max_symbols]
        optimization_results = []
        
        total_combinations = len(target_profits) * len(stop_losses)
        current_combination = 0
        
        for target in target_profits:
            for stop_loss in stop_losses:
                current_combination += 1
                print(f"[{current_combination}/{total_combinations}] Testing TP={target}%, SL={stop_loss}%...", end=" ")
                
                try:
                    # Backtest with current parameters
                    all_trades = []
                    for symbol in test_symbols:
                        trades = self.backtest_symbol(symbol, target, stop_loss)
                        all_trades.extend(trades)
                    
                    if all_trades:
                        stats = self.analyze_results(all_trades)
                        optimization_results.append({
                            'target_pct': target,
                            'stop_loss_pct': stop_loss,
                            'total_trades': stats['total_trades'],
                            'win_rate': stats['win_rate'],
                            'total_profit': stats['total_profit'],
                            'avg_profit': stats['avg_profit'],
                            'target_hits': stats['target_hits'],
                            'stop_losses': stats['stop_losses']
                        })
                        print(f"✓ {stats['total_trades']} trades, {stats['total_profit']:.1f}% profit")
                    else:
                        print("✗ No trades")
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        # Analyze optimization results
        if optimization_results:
            opt_df = pd.DataFrame(optimization_results)
            opt_df = opt_df.sort_values('total_profit', ascending=False)
            
            print("\n" + "=" * 60)
            print("OPTIMIZATION RESULTS")
            print("=" * 60)
            
            print("🏆 TOP 10 PARAMETER COMBINATIONS (by total profit):")
            print(opt_df.head(10)[['target_pct', 'stop_loss_pct', 'total_trades', 'win_rate', 'total_profit', 'avg_profit']].to_string(index=False))
            
            print(f"\n📊 BEST PARAMETERS:")
            best = opt_df.iloc[0]
            print(f"   Target: {best['target_pct']}%")
            print(f"   Stop Loss: {best['stop_loss_pct']}%")
            print(f"   Total Profit: {best['total_profit']:.2f}%")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            print(f"   Total Trades: {best['total_trades']}")
            
            return opt_df
        
        return None

def main():
    print("=== Crypto 1-Hour Candle Strategy Backtester ===")
    print("Strategy: Green candle → LONG, Red candle → SHORT")
    print("Using COMPLETE historical data from launch")
    print("=" * 60)
    
    backtester = CryptoHourlyBacktester()
    
    while True:
        print("\nChoose an option:")
        print("1. Run backtest with default parameters (TP=20%, SL=7%)")
        print("2. Run backtest with custom parameters")
        print("3. Optimize parameters")
        print("4. Show optimization parameter lists")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print(f"\nRunning backtest with default parameters...")
            trades = backtester.backtest_all_symbols()
            stats = backtester.analyze_results(trades)
            backtester.print_results(trades, stats)
            
        elif choice == '2':
            try:
                target = float(input("Enter target profit % (e.g., 20): "))
                stop_loss = float(input("Enter stop loss % (e.g., 7): "))
                
                print(f"\nRunning backtest with TP={target}%, SL={stop_loss}%...")
                trades = backtester.backtest_all_symbols(target, stop_loss)
                stats = backtester.analyze_results(trades)
                backtester.print_results(trades, stats)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                
        elif choice == '3':
            print(f"\nStarting parameter optimization...")
            opt_results = backtester.optimize_parameters()
            
        elif choice == '4':
            print(f"\n📋 OPTIMIZATION PARAMETER LISTS:")
            print(f"Target Profits (%): {TARGET_PROFITS}")
            print(f"Stop Losses (%): {STOP_LOSSES}")
            print(f"Total combinations: {len(TARGET_PROFITS) * len(STOP_LOSSES)}")
            
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
