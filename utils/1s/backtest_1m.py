#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OneMinuteBacktester:
    def __init__(self, data_folder="1m_data", initial_capital=2000.0):
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
        
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv') and '_1m_launch_' in f]
        symbol_data = []
        
        for filename in csv_files:
            try:
                # Extract symbol and launch date from filename like "BTCUSDT_1m_launch_2017-08-17_10-00-00.csv"
                parts = filename.split('_1m_launch_')
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
                symbol = filename.split('_1m_launch_')[0] if '_1m_launch_' in filename else filename
                launch_date = datetime(2099, 1, 1)
                symbol_data.append((symbol, launch_date, filename))
        
        # Sort by launch date (earliest first)
        symbol_data.sort(key=lambda x: x[1])
        
        # Extract just the symbols in chronological order
        symbols = [item[0] for item in symbol_data]
        
        print(f"📊 Found {len(symbols)} symbols with 1-minute launch data")
        print(f"📅 Sorted chronologically by launch date")
        
        if symbol_data:
            earliest = symbol_data[0]
            latest = symbol_data[-1]
            print(f"   Earliest: {earliest[0]} ({earliest[1].strftime('%Y-%m-%d %H:%M')})")
            print(f"   Latest:   {latest[0]} ({latest[1].strftime('%Y-%m-%d %H:%M')})")
        
        return symbols
    
    def filter_symbols_by_time(self, time_range_str):
        """
        Filter symbols by launch time range
        
        Args:
            time_range_str: Time range in format "HH:MM-HH:MM" (e.g., "16:00-23:59")
        
        Returns:
            List of filtered symbols
        """
        if not time_range_str or not self.symbols:
            return self.symbols
        
        try:
            # Parse time range
            start_time_str, end_time_str = time_range_str.split('-')
            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))
            
            print(f"🕐 Filtering coins launched between {start_time_str} and {end_time_str}")
            
            filtered_symbols = []
            total_checked = 0
            
            for symbol in self.symbols:
                try:
                    # Load symbol data to get launch time
                    csv_files = [f for f in os.listdir(self.data_folder) 
                                if f.startswith(f"{symbol}_1m_launch_") and f.endswith('.csv')]
                    
                    if csv_files:
                        total_checked += 1
                        # Extract launch time from filename or CSV
                        filename = csv_files[0]
                        
                        # Try to parse from filename first
                        try:
                            parts = filename.split('_1m_launch_')
                            if len(parts) == 2:
                                date_part = parts[1].replace('.csv', '').replace('-', ':')
                                date_str = date_part.replace('_', ' ')
                                launch_datetime = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                                
                                launch_hour = launch_datetime.hour
                                launch_minute = launch_datetime.minute
                                
                                # Check if launch time is within range
                                launch_time_minutes = launch_hour * 60 + launch_minute
                                start_time_minutes = start_hour * 60 + start_minute
                                end_time_minutes = end_hour * 60 + end_minute
                                
                                # Handle overnight ranges (e.g., 22:00-02:00)
                                if start_time_minutes <= end_time_minutes:
                                    # Same day range
                                    if start_time_minutes <= launch_time_minutes <= end_time_minutes:
                                        filtered_symbols.append(symbol)
                                else:
                                    # Overnight range
                                    if launch_time_minutes >= start_time_minutes or launch_time_minutes <= end_time_minutes:
                                        filtered_symbols.append(symbol)
                        except:
                            # Fallback: read from CSV file
                            filepath = os.path.join(self.data_folder, filename)
                            df = pd.read_csv(filepath)
                            if 'launch_date' in df.columns:
                                launch_date_str = df['launch_date'].iloc[0]
                                launch_datetime = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
                                
                                launch_hour = launch_datetime.hour
                                launch_minute = launch_datetime.minute
                                
                                # Check if launch time is within range
                                launch_time_minutes = launch_hour * 60 + launch_minute
                                start_time_minutes = start_hour * 60 + start_minute
                                end_time_minutes = end_hour * 60 + end_minute
                                
                                # Handle overnight ranges (e.g., 22:00-02:00)
                                if start_time_minutes <= end_time_minutes:
                                    # Same day range
                                    if start_time_minutes <= launch_time_minutes <= end_time_minutes:
                                        filtered_symbols.append(symbol)
                                else:
                                    # Overnight range
                                    if launch_time_minutes >= start_time_minutes or launch_time_minutes <= end_time_minutes:
                                        filtered_symbols.append(symbol)
                except Exception as e:
                    print(f"⚠️  Warning: Could not check launch time for {symbol}: {e}")
                    continue
            
            print(f"📊 Time filter results:")
            print(f"   Total symbols checked: {total_checked}")
            print(f"   Symbols matching time range: {len(filtered_symbols)}")
            print(f"   Filtered out: {total_checked - len(filtered_symbols)}")
            
            if len(filtered_symbols) > 0:
                print(f"   Time filter efficiency: {len(filtered_symbols)/total_checked*100:.1f}%")
            
            return filtered_symbols
            
        except Exception as e:
            print(f"❌ Error parsing time range '{time_range_str}': {e}")
            print("Format should be 'HH:MM-HH:MM' (e.g., '16:00-23:59')")
            return self.symbols
    
    def load_symbol_data(self, symbol):
        """Load 1-minute data for a specific symbol"""
        csv_files = [f for f in os.listdir(self.data_folder) 
                    if f.startswith(f"{symbol}_1m_launch_") and f.endswith('.csv')]
        
        if not csv_files:
            print(f"❌ No data found for {symbol}")
            return None
        
        filename = os.path.join(self.data_folder, csv_files[0])
        
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")
            return None
    
    def simulate_trade(self, df, symbol, target_pct=15.0, stop_loss_pct=5.0, start_candle=2):
        """
        Simulate trading strategy:
        - Enter long at opening price of specified candle (start_candle parameter)
        - TP: target_pct% gain
        - SL: stop_loss_pct% loss
        """
        # Check if we have enough candles (need at least start_candle + 1 for exit)
        if len(df) < start_candle + 1:
            return None
        
        # Entry at opening price of specified candle (start_candle is 1-indexed)
        entry_index = start_candle - 1  # Convert to 0-indexed
        entry_price = df.iloc[entry_index]['open']
        entry_time = df.iloc[entry_index]['timestamp']
        
        # Get launch date from data
        launch_date = df.iloc[0]['launch_date'] if 'launch_date' in df.columns else 'Unknown'
        
        # Calculate target and stop loss prices
        target_price = entry_price * (1 + target_pct / 100)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        
        # Calculate position size based on current capital
        position_size = self.current_capital / entry_price
        
        # Check each candle for exit conditions (including the entry candle itself)
        for i in range(entry_index, len(df)):  # Start from entry candle
            current_candle = df.iloc[i]
            
            # For the entry candle, we need to check if SL hits first (more conservative)
            # since we bought at open, but the low might have occurred before high
            if i == entry_index:
                # Check entry candle: prioritize stop loss if both TP and SL would hit
                if current_candle['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_time = current_candle['timestamp']
                    profit_usd = position_size * (exit_price - entry_price)  # Will be negative
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100
                    
                    trade = {
                        'symbol': symbol,
                        'launch_date': launch_date,
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
                        'duration_minutes': 0  # Same minute exit
                    }
                    
                    # Update capital
                    self.current_capital += profit_usd
                    return trade
                    
                elif current_candle['high'] >= target_price:
                    exit_price = target_price
                    exit_time = current_candle['timestamp']
                    profit_usd = position_size * (exit_price - entry_price)
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100
                    
                    trade = {
                        'symbol': symbol,
                        'launch_date': launch_date,
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
                        'duration_minutes': 0  # Same minute exit
                    }
                    
                    # Update capital
                    self.current_capital += profit_usd
                    return trade
                    
                # If neither TP nor SL hit on entry candle, continue to next candle
                continue
            
            # For subsequent candles, use original logic
            # Check if target hit (high >= target_price)
            if current_candle['high'] >= target_price:
                exit_price = target_price
                exit_time = current_candle['timestamp']
                profit_usd = position_size * (exit_price - entry_price)
                profit_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trade = {
                    'symbol': symbol,
                    'launch_date': launch_date,
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
                    'duration_minutes': i - entry_index  # Minutes from entry
                }
                
                # Update capital
                self.current_capital += profit_usd
                return trade
            
            # Check if stop loss hit (low <= stop_loss_price)
            elif current_candle['low'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_time = current_candle['timestamp']
                profit_usd = position_size * (exit_price - entry_price)  # Will be negative
                profit_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trade = {
                    'symbol': symbol,
                    'launch_date': launch_date,
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
                    'duration_minutes': i - entry_index  # Minutes from entry
                }
                
                # Update capital
                self.current_capital += profit_usd
                return trade
        
        # If neither target nor stop loss hit, exit at last price (end of 4h period)
        last_candle = df.iloc[-1]
        exit_price = last_candle['close']
        exit_time = last_candle['timestamp']
        profit_usd = position_size * (exit_price - entry_price)
        profit_pct = ((exit_price - entry_price) / entry_price) * 100
        
        trade = {
            'symbol': symbol,
            'launch_date': launch_date,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss_price': stop_loss_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'profit_usd': profit_usd,
            'profit_pct': profit_pct,
            'exit_reason': '4h Timeout',
            'capital_before': self.current_capital,
            'capital_after': self.current_capital + profit_usd,
            'duration_minutes': len(df) - 1 - entry_index  # Minutes from entry to end
        }
        
        # Update capital
        self.current_capital += profit_usd
        return trade
    
    def backtest_all_symbols(self, target_pct=15.0, stop_loss_pct=5.0, start_candle=2):
        """Run backtest on all available symbols"""
        print(f"🚀 Starting backtest on {len(self.symbols)} symbols")
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Target: {target_pct}% | 🛑 Stop Loss: {stop_loss_pct}%")
        print(f"📊 Strategy: Long at candle #{start_candle} open")
        print("=" * 80)
        
        target_hits = 0
        stop_losses = 0
        timeouts = 0
        successful_symbols = 0
        failed_symbols = 0
        
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
            trade = self.simulate_trade(df, symbol, target_pct, stop_loss_pct, start_candle)
            
            if trade:
                self.all_trades.append(trade)
                
                # Count exit reasons
                if trade['exit_reason'] == 'Target Hit':
                    target_hits += 1
                    print(f"🎯 TP: ${trade['profit_usd']:+7.0f} | Capital: ${self.current_capital:,.0f}")
                elif trade['exit_reason'] == 'Stop Loss':
                    stop_losses += 1
                    print(f"🛑 SL: ${trade['profit_usd']:+7.0f} | Capital: ${self.current_capital:,.0f}")
                else:
                    timeouts += 1
                    print(f"⏰ TO: ${trade['profit_usd']:+7.0f} | Capital: ${self.current_capital:,.0f}")
                
                successful_symbols += 1
            else:
                print("❌ Trade simulation failed")
                failed_symbols += 1
        
        # Final statistics
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 80)
        
        total_trades = len(self.all_trades)
        total_profit = self.current_capital - self.initial_capital
        total_return = (total_profit / self.initial_capital) * 100
        
        print(f"💰 Portfolio Performance:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Capital: ${self.current_capital:,.2f}")
        print(f"   Total Profit/Loss: ${total_profit:+,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        
        print(f"\n📈 Trade Analysis:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Successful Symbols: {successful_symbols}")
        print(f"   Failed Symbols: {failed_symbols}")
        
        if total_trades > 0:
            print(f"\n🎯 Exit Reasons:")
            print(f"   Target Hits: {target_hits} ({target_hits/total_trades*100:.1f}%)")
            print(f"   Stop Losses: {stop_losses} ({stop_losses/total_trades*100:.1f}%)")
            print(f"   Timeouts: {timeouts} ({timeouts/total_trades*100:.1f}%)")
            
            # Calculate win rate (target hits / total trades)
            win_rate = (target_hits / total_trades) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
            
            # Average profit per trade
            avg_profit = total_profit / total_trades
            print(f"   Average Profit/Trade: ${avg_profit:+.2f}")
        
        return {
            'total_trades': total_trades,
            'target_hits': target_hits,
            'stop_losses': stop_losses,
            'timeouts': timeouts,
            'win_rate': win_rate if total_trades > 0 else 0,
            'total_profit': total_profit,
            'total_return': total_return,
            'final_capital': self.current_capital
        }
    
    def save_detailed_results(self):
        """Save detailed trade results to CSV"""
        if not self.all_trades:
            print("No trades to save")
            return
        
        df = pd.DataFrame(self.all_trades)
        
        # Format datetime columns
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Round numerical columns
        numerical_cols = ['entry_price', 'target_price', 'stop_loss_price', 'exit_price', 'position_size', 'profit_usd', 'profit_pct', 
                         'capital_before', 'capital_after']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_1m_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        # Show sample trades
        print(f"\n📋 Sample Trades (First 5):")
        sample_cols = ['symbol', 'launch_date', 'entry_price', 'target_price', 'stop_loss_price', 'exit_price', 'profit_usd', 'exit_reason', 'capital_after']
        sample_df = df[sample_cols].head().copy()
        sample_df['launch_date'] = sample_df['launch_date'].str[:10]  # Show only date part
        
        # Round price columns for better display
        price_cols = ['entry_price', 'target_price', 'stop_loss_price', 'exit_price']
        for col in price_cols:
            sample_df[col] = sample_df[col].round(6)
        
        print(sample_df.to_string(index=False))
        
        return filename
    
    def analyze_performance(self):
        """Analyze performance by different metrics"""
        if not self.all_trades:
            return
        
        df = pd.DataFrame(self.all_trades)
        
        print(f"\n📊 DETAILED PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Performance by exit reason
        print(f"💰 Profit by Exit Reason:")
        for reason in ['Target Hit', 'Stop Loss', '4h Timeout']:
            reason_trades = df[df['exit_reason'] == reason]
            if len(reason_trades) > 0:
                total_profit = reason_trades['profit_usd'].sum()
                avg_profit = reason_trades['profit_usd'].mean()
                count = len(reason_trades)
                print(f"   {reason:<12}: {count:3d} trades | Total: ${total_profit:+8.0f} | Avg: ${avg_profit:+6.0f}")
        
        # Best and worst trades
        print(f"\n🏆 Best Trades (Top 5):")
        best_trades = df.nlargest(5, 'profit_usd')[['symbol', 'profit_usd', 'profit_pct', 'exit_reason']]
        print(best_trades.to_string(index=False))
        
        print(f"\n📉 Worst Trades (Bottom 5):")
        worst_trades = df.nsmallest(5, 'profit_usd')[['symbol', 'profit_usd', 'profit_pct', 'exit_reason']]
        print(worst_trades.to_string(index=False))
        
        # Duration analysis
        print(f"\n⏱️  Duration Analysis:")
        avg_duration = df['duration_minutes'].mean()
        print(f"   Average trade duration: {avg_duration:.1f} minutes")
        
        duration_by_reason = df.groupby('exit_reason')['duration_minutes'].mean()
        for reason, avg_dur in duration_by_reason.items():
            print(f"   {reason:<12}: {avg_dur:.1f} minutes")
    
    def grid_search_optimization(self, tp_range, sl_range, max_symbols=None):
        """
        Perform grid search to find optimal TP and SL parameters
        tp_range: range of target profit percentages to test
        sl_range: range of stop loss percentages to test
        max_symbols: limit symbols for faster testing (None = use all)
        """
        print(f"\n🔍 === GRID SEARCH OPTIMIZATION ===")
        print(f"🎯 Testing TP range: {tp_range}")
        print(f"🛑 Testing SL range: {sl_range}")
        print(f"🔢 Total combinations: {len(tp_range) * len(sl_range)}")
        
        if max_symbols:
            test_symbols = self.symbols[:max_symbols]
            print(f"📊 Using first {max_symbols} symbols for speed")
        else:
            test_symbols = self.symbols
            print(f"📊 Using all {len(test_symbols)} symbols")
        
        print("=" * 80)
        
        optimization_results = []
        total_combinations = len(tp_range) * len(sl_range)
        current_combination = 0
        
        for tp in tp_range:
            for sl in sl_range:
                current_combination += 1
                print(f"[{current_combination:3d}/{total_combinations}] TP={tp:2d}%, SL={sl:2d}%...", end=" ")
                
                try:
                    # Reset for this parameter combination (don't create new instance)
                    original_capital = self.current_capital
                    self.current_capital = self.initial_capital
                    self.all_trades = []
                    
                    # Use the same symbol list consistently
                    symbols_to_test = test_symbols
                    
                    # Run backtest with current parameters (quietly)
                    target_hits_temp = 0
                    stop_losses_temp = 0
                    timeouts_temp = 0
                    trades_temp = []
                    
                    for symbol in symbols_to_test:
                        # Load data for symbol
                        df = self.load_symbol_data(symbol)
                        
                        if df is None or len(df) < 2:
                            continue
                        
                        # Run trade simulation (using default start_candle for grid search)
                        trade = self.simulate_trade(df, symbol, tp, sl, start_candle=2)
                        
                        if trade:
                            trades_temp.append(trade)
                            
                            # Count exit reasons
                            if trade['exit_reason'] == 'Target Hit':
                                target_hits_temp += 1
                            elif trade['exit_reason'] == 'Stop Loss':
                                stop_losses_temp += 1
                            else:
                                timeouts_temp += 1
                    
                    # Calculate results
                    if trades_temp:
                        total_profit = self.current_capital - self.initial_capital
                        total_return = (total_profit / self.initial_capital) * 100
                        total_trades = len(trades_temp)
                        
                        # Use the counts we already calculated
                        target_hits = target_hits_temp
                        stop_losses = stop_losses_temp
                        timeouts = timeouts_temp
                        
                        win_rate = (target_hits / total_trades) * 100 if total_trades > 0 else 0
                        
                        optimization_results.append({
                            'tp_pct': tp,
                            'sl_pct': sl,
                            'total_profit_usd': total_profit,
                            'total_return_pct': total_return,
                            'total_trades': total_trades,
                            'target_hits': target_hits,
                            'stop_losses': stop_losses,
                            'timeouts': timeouts,
                            'win_rate': win_rate,
                            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
                            'final_capital': self.current_capital
                        })
                        
                        print(f"💰 ${total_profit:+8.0f} ({total_return:+5.1f}%) | WR: {win_rate:4.1f}%")
                    else:
                        print("❌ No trades")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                # Restore original capital state
                self.current_capital = original_capital
        
        # Analyze optimization results
        if optimization_results:
            opt_df = pd.DataFrame(optimization_results)
            opt_df = opt_df.sort_values('total_profit_usd', ascending=False)
            
            print("\n" + "=" * 80)
            print("🏆 GRID SEARCH RESULTS")
            print("=" * 80)
            
            print("🥇 TOP 10 PARAMETER COMBINATIONS (by total profit USD):")
            print(f"{'Rank':<4} {'TP%':<4} {'SL%':<4} {'Profit USD':<12} {'Return%':<8} {'WinRate%':<9} {'TH':<3} {'SL':<3} {'TO':<3}")
            print("-" * 70)
            for i, row in enumerate(opt_df.head(10).itertuples(), 1):
                print(f"{i:<4} {row.tp_pct:<4} {row.sl_pct:<4} ${row.total_profit_usd:<11,.0f} {row.total_return_pct:<7.1f}% {row.win_rate:<8.1f}% {row.target_hits:<3} {row.stop_losses:<3} {row.timeouts:<3}")
            
            print(f"\n🎯 BEST PARAMETERS:")
            best = opt_df.iloc[0]
            print(f"   🎯 Optimal TP: {best['tp_pct']}%")
            print(f"   🛑 Optimal SL: {best['sl_pct']}%")
            print(f"   💰 Maximum Profit: ${best['total_profit_usd']:,.2f}")
            print(f"   📈 Maximum Return: {best['total_return_pct']:+.2f}%")
            print(f"   🎯 Win Rate: {best['win_rate']:.1f}%")
            print(f"   📊 Trades: {best['total_trades']} total")
            print(f"   ✅ Target Hits: {best['target_hits']}")
            print(f"   ❌ Stop Losses: {best['stop_losses']}")
            print(f"   ⏰ Timeouts: {best['timeouts']}")
            
            # Save optimization results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            opt_filename = f"grid_search_optimization_{timestamp}.csv"
            opt_df.to_csv(opt_filename, index=False)
            print(f"\n💾 Grid search results saved to: {opt_filename}")
            
            # Show heatmap summary
            print(f"\n📊 PERFORMANCE HEATMAP SUMMARY:")
            print(f"   TP range tested: {min(tp_range)}% - {max(tp_range)}%")
            print(f"   SL range tested: {min(sl_range)}% - {max(sl_range)}%")
            print(f"   Best profit: ${opt_df['total_profit_usd'].max():,.0f}")
            print(f"   Worst result: ${opt_df['total_profit_usd'].min():,.0f}")
            print(f"   Profitable combinations: {len(opt_df[opt_df['total_profit_usd'] > 0])}/{len(opt_df)}")
            
            return opt_df
        
        return None

def main():
    print("🚀 === 1-Minute Launch Data Backtester ===")
    print("Strategy: Long at 2nd candle open")
    print("Capital: $2000 compounded")
    print("=" * 80)
    
    # ===== CONFIGURABLE PARAMETERS =====
    DEFAULT_TP = 20.0        # Default Target Profit %
    DEFAULT_SL = 10.0         # Default Stop Loss %
    DEFAULT_START_CANDLE = 2 # Default candle to enter (2=second candle, 3=third candle, etc.)
    LAST_COINS = 10         # Number of coins to test (100, 200, 300, etc.) - takes last N from chronologically sorted list (most recent)
    TIME_FILTER = False      # Filter coins by launch time (True/False)
    TIME_RANGE = "16:00-23:59" # Time range to filter (format: "HH:MM-HH:MM" in 24-hour format)
    # ====================================
    
    # Initialize backtester
    backtester = OneMinuteBacktester(data_folder="1m_data", initial_capital=2000.0)
    
    if not backtester.symbols:
        print("❌ No 1-minute data found! Please run price_downloader.py first.")
        return
    
    # Limit to LAST_COINS from chronologically sorted list (most recently launched)
    if LAST_COINS and LAST_COINS < len(backtester.symbols):
        original_count = len(backtester.symbols)
        backtester.symbols = backtester.symbols[-LAST_COINS:]  # Take last N coins
        print(f"🎯 Limited to last {LAST_COINS} coins (out of {original_count} available) - most recently launched")
    else:
        print(f"🎯 Using all {len(backtester.symbols)} available coins")
    
    # Apply time filter if enabled
    if TIME_FILTER and TIME_RANGE:
        print(f"\n🕐 Time filter enabled: {TIME_RANGE}")
        original_symbol_count = len(backtester.symbols)
        backtester.symbols = backtester.filter_symbols_by_time(TIME_RANGE)
        
        if len(backtester.symbols) == 0:
            print("❌ No coins found matching the time filter criteria!")
            print("💡 Try adjusting TIME_RANGE or set TIME_FILTER = False")
            return
        
        print(f"✅ Time filter applied: {len(backtester.symbols)} coins remain (filtered out {original_symbol_count - len(backtester.symbols)})")
    else:
        print(f"⏰ Time filter disabled - testing all hours")
    
    while True:
        print("\nChoose an option:")
        print(f"1. 🎯 Run single backtest (TP={DEFAULT_TP}%, SL={DEFAULT_SL}%, Candle={DEFAULT_START_CANDLE}, Coins={len(backtester.symbols)})")
        print("2. ⚙️  Run custom single backtest (specify your own TP/SL/Candle)")
        print("3. 🔍 Grid search optimization (find best TP/SL)")
        print("4. 🚀 Quick grid search (first 20 coins only)")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print(f"\n🚀 Running single backtest with TP={DEFAULT_TP}%, SL={DEFAULT_SL}%, Entry Candle={DEFAULT_START_CANDLE}...")
            
            # Run single backtest with default parameters
            results = backtester.backtest_all_symbols(target_pct=DEFAULT_TP, stop_loss_pct=DEFAULT_SL, start_candle=DEFAULT_START_CANDLE)
            
            # Analyze performance
            backtester.analyze_performance()
            
            # Save detailed results
            backtester.save_detailed_results()
            
            print(f"\n✅ Backtest complete!")
            print(f"🎯 Final Result: ${results['total_profit']:+,.2f} ({results['total_return']:+.2f}%)")
            
        elif choice == '2':
            try:
                custom_tp = float(input(f"Enter Target Profit % (current default: {DEFAULT_TP}%): "))
                custom_sl = float(input(f"Enter Stop Loss % (current default: {DEFAULT_SL}%): "))
                custom_candle = int(input(f"Enter Entry Candle # (current default: {DEFAULT_START_CANDLE}): "))
                
                print(f"\n🚀 Running custom backtest with TP={custom_tp}%, SL={custom_sl}%, Entry Candle={custom_candle}...")
                
                # Reset backtester state
                backtester.current_capital = backtester.initial_capital
                backtester.all_trades = []
                
                # Run single backtest with custom parameters
                results = backtester.backtest_all_symbols(target_pct=custom_tp, stop_loss_pct=custom_sl, start_candle=custom_candle)
                
                # Analyze performance
                backtester.analyze_performance()
                
                # Save detailed results
                backtester.save_detailed_results()
                
                print(f"\n✅ Custom backtest complete!")
                print(f"🎯 Final Result: ${results['total_profit']:+,.2f} ({results['total_return']:+.2f}%)")
                
            except ValueError:
                print("❌ Invalid input. Please enter numeric values.")
            
        elif choice == '3':
            print(f"\n🔍 Starting FULL grid search optimization...")
            print("⚠️  This will test 35 combinations and may take 5-10 minutes")
            
            confirm = input("Continue with full optimization? (y/N): ").strip().lower()
            if confirm == 'y':
                # Define parameter ranges (your custom ranges)
                tp_range = [10, 15, 20, 30, 45, 60, 80]  # 10% to 80% in steps of 10 (8 values)
                sl_range =   [2, 5, 7, 10, 15] # 1% to 15% in steps of 2 (8 values)          # All SL values 1-15
                    
                print(f"🎯 TP range: {tp_range}")
                print(f"🛑 SL range: {sl_range}")
                
                # Run grid search on all symbols
                opt_results = backtester.grid_search_optimization(tp_range, sl_range, max_symbols=None)
                
                if opt_results is not None:
                    print(f"\n🎯 OPTIMIZATION COMPLETE!")
                    best = opt_results.iloc[0]
                    print(f"🏆 Best combination: TP={best['tp_pct']}%, SL={best['sl_pct']}%")
                    print(f"💰 Maximum profit: ${best['total_profit_usd']:+,.2f}")
            else:
                print("Grid search cancelled.")
                
        elif choice == '4':
            print(f"\n🚀 Starting QUICK grid search optimization...")
            print("📊 Testing on first 20 coins for speed")
            
            # Define parameter ranges  
            tp_range = [10, 15, 20, 30, 45, 60, 80]  # 10% to 80% in steps of 10 (8 values)
            sl_range =   [2, 5, 7, 10, 15] # 1% to 15% in steps of 2 (8 values)
            
            print(f"🎯 TP range: {tp_range}")
            print(f"🛑 SL range: {sl_range}")
            print(f"🔢 Testing {len(tp_range) * len(sl_range)} combinations")
            
            # Run grid search on subset
            opt_results = backtester.grid_search_optimization(tp_range, sl_range, max_symbols=20)
            
            if opt_results is not None:
                print(f"\n🎯 QUICK OPTIMIZATION COMPLETE!")
                best = opt_results.iloc[0]
                print(f"🏆 Best combination: TP={best['tp_pct']}%, SL={best['sl_pct']}%")
                print(f"💰 Maximum profit: ${best['total_profit_usd']:+,.2f}")
                print(f"\n💡 Note: This is based on first 20 coins only. Run full optimization for complete results.")
                
        elif choice == '5':
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
