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
DEFAULT_TARGET = 10  # 20%
DEFAULT_STOP_LOSS = 3  # 7%

class CryptoHourlyBacktester:
    def __init__(self, data_folder="launch_day_data"):
        self.data_folder = data_folder
        self.initial_capital = 2000.0  # Starting with $2000
        self.current_capital = self.initial_capital
        self.futures_list = self._load_futures_list()
        self.results = []
        self.coin_results = {}  # Track per-coin results
        
    def _load_futures_list(self):
        """Load the list of available launch day data files"""
        if not os.path.exists(self.data_folder):
            print(f"Warning: Data folder {self.data_folder} not found!")
            return []
        
        # Get all CSV files from launch_day_data folder
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv') and 'analysis' not in f]
        
        # Extract symbol names from filenames
        symbols = []
        for filename in csv_files:
            # Format: SYMBOL_launch_DATE.csv
            symbol = filename.split('_launch_')[0]
            symbols.append(symbol)
        
        print(f"Found {len(symbols)} coins with launch day data")
        return sorted(symbols)
    
    def load_symbol_data(self, symbol):
        """Load launch day price data for a specific symbol"""
        # Find the launch day CSV file for this symbol
        csv_files = [f for f in os.listdir(self.data_folder) if f.startswith(f"{symbol}_launch_") and f.endswith('.csv')]
        
        if not csv_files:
            print(f"Warning: Launch day data not found for {symbol}")
            return None
        
        filename = os.path.join(self.data_folder, csv_files[0])
        
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
    
    def simulate_trade(self, df, entry_idx, trade_type, target_pct, stop_loss_pct, capital_amount):
        """
        Simulate a single trade with capital management
        trade_type: 'long' for green candle, 'short' for red candle
        target_pct: target profit percentage
        stop_loss_pct: stop loss percentage
        capital_amount: USD amount to trade
        """
        if entry_idx >= len(df) - 1:
            return None
        
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row['close']  # Enter at close of signal candle
        entry_time = entry_row['timestamp']
        
        # Calculate position size based on capital
        position_size = capital_amount / entry_price
        
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
                    profit_usd = position_size * (target_price - entry_price)
                    profit_pct = ((target_price - entry_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_usd,
                        'capital_used': capital_amount,
                        'position_size': position_size,
                        'exit_reason': 'Target Hit',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
                
                # Check stop loss hit (price went down)
                if current_row['low'] <= stop_loss_price:
                    profit_usd = position_size * (stop_loss_price - entry_price)
                    profit_pct = ((stop_loss_price - entry_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_usd,
                        'capital_used': capital_amount,
                        'position_size': position_size,
                        'exit_reason': 'Stop Loss',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
            
            else:  # 'short'
                # Check target hit (price went down)
                if current_row['low'] <= target_price:
                    profit_usd = position_size * (entry_price - target_price)
                    profit_pct = ((entry_price - target_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_usd,
                        'capital_used': capital_amount,
                        'position_size': position_size,
                        'exit_reason': 'Target Hit',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
                
                # Check stop loss hit (price went up)
                if current_row['high'] >= stop_loss_price:
                    profit_usd = position_size * (entry_price - stop_loss_price)
                    profit_pct = ((entry_price - stop_loss_price) / entry_price) * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss_price,
                        'trade_type': trade_type,
                        'profit_pct': profit_pct,
                        'profit_usd': profit_usd,
                        'capital_used': capital_amount,
                        'position_size': position_size,
                        'exit_reason': 'Stop Loss',
                        'duration_hours': (current_time - entry_time).total_seconds() / 3600,
                        'target_pct': target_pct,
                        'stop_loss_pct': stop_loss_pct
                    }
        
        # If no exit condition met, exit at last available price
        last_row = df.iloc[-1]
        if trade_type == 'long':
            profit_usd = position_size * (last_row['close'] - entry_price)
            profit_pct = ((last_row['close'] - entry_price) / entry_price) * 100
        else:
            profit_usd = position_size * (entry_price - last_row['close'])
            profit_pct = ((entry_price - last_row['close']) / entry_price) * 100
        
        return {
            'entry_time': entry_time,
            'exit_time': last_row['timestamp'],
            'entry_price': entry_price,
            'exit_price': last_row['close'],
            'trade_type': trade_type,
            'profit_pct': profit_pct,
            'profit_usd': profit_usd,
            'capital_used': capital_amount,
            'position_size': position_size,
            'exit_reason': 'End of Data',
            'duration_hours': (last_row['timestamp'] - entry_time).total_seconds() / 3600,
            'target_pct': target_pct,
            'stop_loss_pct': stop_loss_pct
        }
    
    def backtest_symbol_with_capital(self, symbol, target_pct=DEFAULT_TARGET, stop_loss_pct=DEFAULT_STOP_LOSS):
        """Backtest a single symbol with capital management - start with $2000"""
        df = self.load_symbol_data(symbol)
        if df is None or len(df) < 2:
            return None
        
        # Start with initial capital for this coin
        capital = self.initial_capital  # $2000 per coin
        
        trades = []
        coin_profitable = False
        target_hits = 0
        stop_losses = 0
        
        # Only take one trade per coin (first signal)
        for i in range(len(df) - 1):
            current_candle = df.iloc[i]
            
            if current_candle['candle_type'] == 'green':
                # Green candle -> go LONG
                trade = self.simulate_trade(df, i, 'long', target_pct, stop_loss_pct, capital)
                if trade:
                    trade['symbol'] = symbol
                    trades.append(trade)
                    
                    # Update metrics
                    if trade['exit_reason'] == 'Target Hit':
                        coin_profitable = True
                        target_hits += 1
                    elif trade['exit_reason'] == 'Stop Loss':
                        stop_losses += 1
                    
                    break  # Only one trade per coin
            
            elif current_candle['candle_type'] == 'red':
                # Red candle -> go SHORT
                trade = self.simulate_trade(df, i, 'short', target_pct, stop_loss_pct, capital)
                if trade:
                    trade['symbol'] = symbol
                    trades.append(trade)
                    
                    # Update metrics
                    if trade['exit_reason'] == 'Target Hit':
                        coin_profitable = True
                        target_hits += 1
                    elif trade['exit_reason'] == 'Stop Loss':
                        stop_losses += 1
                    
                    break  # Only one trade per coin
        
        # Return coin results
        if trades:
            total_profit_usd = sum([t['profit_usd'] for t in trades])
            return {
                'symbol': symbol,
                'trades': trades,
                'profitable': coin_profitable,
                'target_hits': target_hits,
                'stop_losses': stop_losses,
                'total_profit_usd': total_profit_usd,
                'final_capital': capital + total_profit_usd
            }
        
        return None
    
    def backtest_all_symbols(self, target_pct=DEFAULT_TARGET, stop_loss_pct=DEFAULT_STOP_LOSS):
        """Backtest all symbols with capital management"""
        print(f"\n🚀 === LAUNCH DAY TRADING BACKTEST ===")
        print(f"💰 Starting Capital: ${self.initial_capital:,.2f} per coin")
        print(f"📊 Strategy: Green candle = LONG, Red candle = SHORT")
        print(f"🎯 Target: {target_pct}%, Stop Loss: {stop_loss_pct}%")
        print(f"🪙 Testing {len(self.futures_list)} coins")
        print("=" * 80)
        
        all_trades = []
        coin_results = []
        profitable_coins = 0
        total_target_hits = 0
        total_stop_losses = 0
        
        for i, symbol in enumerate(self.futures_list, 1):
            print(f"[{i:3d}/{len(self.futures_list)}] {symbol:12s}...", end=" ")
            
            try:
                result = self.backtest_symbol_with_capital(symbol, target_pct, stop_loss_pct)
                if result:
                    all_trades.extend(result['trades'])
                    coin_results.append(result)
                    
                    if result['profitable']:
                        profitable_coins += 1
                    
                    total_target_hits += result['target_hits']
                    total_stop_losses += result['stop_losses']
                    
                    # Show result
                    profit_usd = result['total_profit_usd']
                    if result['profitable']:
                        print(f"✅ PROFIT: ${profit_usd:+7.2f}")
                    else:
                        print(f"❌ LOSS: ${profit_usd:+7.2f}")
                else:
                    print("⚪ No signals")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Store results
        self.coin_results = coin_results
        
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        print(f"💰 Total coins with data: {len(coin_results)}")
        print(f"✅ Profitable coins: {profitable_coins}")
        print(f"❌ Loss-making coins: {len(coin_results) - profitable_coins}")
        print(f"🎯 Total target hits: {total_target_hits}")
        print(f"🛑 Total stop losses: {total_stop_losses}")
        print(f"📈 Success rate: {(profitable_coins / len(coin_results) * 100):.1f}%" if coin_results else "0%")
        
        return all_trades
    
    def analyze_capital_results(self):
        """Analyze capital-based backtest results"""
        if not self.coin_results:
            print("No coin results to analyze")
            return {}
        
        # Calculate portfolio statistics
        total_initial_capital = len(self.coin_results) * self.initial_capital
        total_final_capital = sum([r['final_capital'] for r in self.coin_results])
        total_profit_usd = total_final_capital - total_initial_capital
        
        profitable_coins = sum([1 for r in self.coin_results if r['profitable']])
        total_target_hits = sum([r['target_hits'] for r in self.coin_results])
        total_stop_losses = sum([r['stop_losses'] for r in self.coin_results])
        
        # Best and worst performers
        best_performer = max(self.coin_results, key=lambda x: x['total_profit_usd'])
        worst_performer = min(self.coin_results, key=lambda x: x['total_profit_usd'])
        
        stats = {
            'total_coins': len(self.coin_results),
            'profitable_coins': profitable_coins,
            'losing_coins': len(self.coin_results) - profitable_coins,
            'success_rate': (profitable_coins / len(self.coin_results)) * 100,
            'total_initial_capital': total_initial_capital,
            'total_final_capital': total_final_capital,
            'total_profit_usd': total_profit_usd,
            'profit_percentage': (total_profit_usd / total_initial_capital) * 100,
            'target_hits': total_target_hits,
            'stop_losses': total_stop_losses,
            'best_performer': best_performer,
            'worst_performer': worst_performer
        }
        
        return stats
    
    def print_capital_results(self):
        """Print detailed capital-based results"""
        stats = self.analyze_capital_results()
        if not stats:
            return
        
        print("\n" + "=" * 80)
        print("💰 CAPITAL MANAGEMENT BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"🏦 PORTFOLIO SUMMARY:")
        print(f"   💵 Total Initial Capital: ${stats['total_initial_capital']:,.2f}")
        print(f"   💰 Total Final Capital: ${stats['total_final_capital']:,.2f}")
        print(f"   📊 Total Profit/Loss: ${stats['total_profit_usd']:+,.2f}")
        print(f"   📈 Portfolio Return: {stats['profit_percentage']:+.2f}%")
        
        print(f"\n🪙 COIN ANALYSIS:")
        print(f"   ✅ Profitable Coins: {stats['profitable_coins']} / {stats['total_coins']}")
        print(f"   ❌ Loss-making Coins: {stats['losing_coins']} / {stats['total_coins']}")
        print(f"   🎯 Success Rate: {stats['success_rate']:.1f}%")
        
        print(f"\n🎯 TRADE OUTCOMES:")
        print(f"   ✅ Target Hits: {stats['target_hits']}")
        print(f"   ❌ Stop Losses: {stats['stop_losses']}")
        print(f"   📊 Target Hit Rate: {(stats['target_hits'] / (stats['target_hits'] + stats['stop_losses']) * 100):.1f}%" if (stats['target_hits'] + stats['stop_losses']) > 0 else "0%")
        
        print(f"\n🏆 BEST PERFORMER:")
        best = stats['best_performer']
        print(f"   🪙 Symbol: {best['symbol']}")
        print(f"   💰 Profit: ${best['total_profit_usd']:+.2f}")
        
        print(f"\n📉 WORST PERFORMER:")
        worst = stats['worst_performer']
        print(f"   🪙 Symbol: {worst['symbol']}")
        print(f"   💸 Loss: ${worst['total_profit_usd']:+.2f}")
        
        # Show top 10 and bottom 10 performers
        sorted_results = sorted(self.coin_results, key=lambda x: x['total_profit_usd'], reverse=True)
        
        print(f"\n🏆 TOP 10 PROFITABLE COINS:")
        print(f"{'Rank':<4} {'Symbol':<12} {'Profit':<10} {'Type':<6} {'Exit':<12}")
        print("-" * 50)
        for i, result in enumerate(sorted_results[:10], 1):
            trade = result['trades'][0] if result['trades'] else None
            trade_type = trade['trade_type'].upper() if trade else "N/A"
            exit_reason = trade['exit_reason'] if trade else "N/A"
            print(f"{i:<4} {result['symbol']:<12} ${result['total_profit_usd']:+7.2f} {trade_type:<6} {exit_reason:<12}")
        
        print(f"\n📉 BOTTOM 10 COINS:")
        print(f"{'Rank':<4} {'Symbol':<12} {'Loss':<10} {'Type':<6} {'Exit':<12}")
        print("-" * 50)
        for i, result in enumerate(sorted_results[-10:], 1):
            trade = result['trades'][0] if result['trades'] else None
            trade_type = trade['trade_type'].upper() if trade else "N/A"
            exit_reason = trade['exit_reason'] if trade else "N/A"
            print(f"{i:<4} {result['symbol']:<12} ${result['total_profit_usd']:+7.2f} {trade_type:<6} {exit_reason:<12}")
        
        # Save detailed results to CSV
        self.save_results_to_csv()
    
    def save_results_to_csv(self):
        """Save detailed results to CSV file"""
        if not self.coin_results:
            return
        
        # Prepare data for CSV
        csv_data = []
        for result in self.coin_results:
            trade = result['trades'][0] if result['trades'] else None
            if trade:
                csv_data.append({
                    'symbol': result['symbol'],
                    'profitable': result['profitable'],
                    'total_profit_usd': result['total_profit_usd'],
                    'final_capital': result['final_capital'],
                    'trade_type': trade['trade_type'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'profit_pct': trade['profit_pct'],
                    'exit_reason': trade['exit_reason'],
                    'duration_hours': trade['duration_hours'],
                    'entry_time': trade['entry_time'],
                    'exit_time': trade['exit_time']
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            filename = f"launch_day_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"\n💾 Detailed results saved to: {filename}")
    
    def grid_search_optimization(self):
        """
        Grid search optimization for TP and SL parameters
        Tests all combinations on all available coins
        """
        print(f"\n🔍 === GRID SEARCH PARAMETER OPTIMIZATION ===")
        print(f"🎯 Target Profits: {TARGET_PROFITS}")
        print(f"🛑 Stop Losses: {STOP_LOSSES}")
        print(f"🔢 Total combinations: {len(TARGET_PROFITS) * len(STOP_LOSSES)}")
        print(f"🪙 Testing on {len(self.futures_list)} coins")
        print("=" * 80)
        
        optimization_results = []
        total_combinations = len(TARGET_PROFITS) * len(STOP_LOSSES)
        current_combination = 0
        
        for target in TARGET_PROFITS:
            for stop_loss in STOP_LOSSES:
                current_combination += 1
                print(f"[{current_combination:2d}/{total_combinations}] TP={target:2d}%, SL={stop_loss:2d}%...", end=" ")
                
                try:
                    # Reset for each parameter combination
                    backtester_temp = CryptoHourlyBacktester()
                    trades = backtester_temp.backtest_all_symbols(target, stop_loss)
                    
                    if backtester_temp.coin_results:
                        stats = backtester_temp.analyze_capital_results()
                        optimization_results.append({
                            'target_pct': target,
                            'stop_loss_pct': stop_loss,
                            'total_profit_usd': stats['total_profit_usd'],
                            'profit_percentage': stats['profit_percentage'],
                            'success_rate': stats['success_rate'],
                            'profitable_coins': stats['profitable_coins'],
                            'target_hits': stats['target_hits'],
                            'stop_losses': stats['stop_losses']
                        })
                        print(f"💰 ${stats['total_profit_usd']:+8.0f} ({stats['profit_percentage']:+5.1f}%) | Success: {stats['success_rate']:4.1f}%")
                    else:
                        print("❌ No results")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # Analyze optimization results
        if optimization_results:
            opt_df = pd.DataFrame(optimization_results)
            opt_df = opt_df.sort_values('total_profit_usd', ascending=False)
            
            print("\n" + "=" * 80)
            print("🏆 GRID SEARCH OPTIMIZATION RESULTS")
            print("=" * 80)
            
            print("🥇 TOP 10 PARAMETER COMBINATIONS (by total profit USD):")
            print(f"{'Rank':<4} {'TP%':<4} {'SL%':<4} {'Profit USD':<12} {'Return%':<8} {'Success%':<9} {'TH':<3} {'SL':<3}")
            print("-" * 60)
            for i, row in enumerate(opt_df.head(10).itertuples(), 1):
                print(f"{i:<4} {row.target_pct:<4} {row.stop_loss_pct:<4} ${row.total_profit_usd:<11,.0f} {row.profit_percentage:<7.1f}% {row.success_rate:<8.1f}% {row.target_hits:<3} {row.stop_losses:<3}")
            
            print(f"\n🎯 BEST PARAMETERS:")
            best = opt_df.iloc[0]
            print(f"   🎯 Target Profit: {best['target_pct']}%")
            print(f"   🛑 Stop Loss: {best['stop_loss_pct']}%")
            print(f"   💰 Total Profit: ${best['total_profit_usd']:,.2f}")
            print(f"   📈 Portfolio Return: {best['profit_percentage']:+.2f}%")
            print(f"   🎯 Success Rate: {best['success_rate']:.1f}%")
            print(f"   ✅ Target Hits: {best['target_hits']}")
            print(f"   ❌ Stop Losses: {best['stop_losses']}")
            
            # Save optimization results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            opt_filename = f"grid_search_optimization_{timestamp}.csv"
            opt_df.to_csv(opt_filename, index=False)
            print(f"\n💾 Optimization results saved to: {opt_filename}")
            
            return opt_df
        
        return None

def main():
    print("🚀 === Launch Day Trading Backtester ===")
    print("💰 Strategy: Green candle → LONG, Red candle → SHORT")
    print("🏦 Capital Management: $2000 per coin")
    print("📊 Data: 24-hour launch day candlesticks")
    print("=" * 80)
    
    backtester = CryptoHourlyBacktester()
    
    if not backtester.futures_list:
        print("❌ No launch day data found! Please run download_launch_day_data.py first.")
        return
    
    while True:
        print("\nChoose an option:")
        print(f"1. 🎯 Run backtest with fixed parameters (TP={DEFAULT_TARGET}%, SL={DEFAULT_STOP_LOSS}%)")
        print("2. 🔍 Grid search optimization (test all TP/SL combinations)")
        print("3. 🚪 Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            print(f"\n🚀 Running backtest with fixed parameters TP={DEFAULT_TARGET}%, SL={DEFAULT_STOP_LOSS}%...")
            trades = backtester.backtest_all_symbols()
            backtester.print_capital_results()
            
        elif choice == '2':
            confirm = input(f"\n⚠️  This will test {len(TARGET_PROFITS) * len(STOP_LOSSES)} combinations. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                backtester.grid_search_optimization()
            else:
                print("Grid search cancelled.")
                
        elif choice == '3':
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
