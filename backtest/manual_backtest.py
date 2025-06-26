#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Backtesting for New Listing Strategy
Simulates trading with predefined symbols and synthetic data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import csv

class ManualBacktest:
    def __init__(self):
        """Initialize the manual backtesting system"""
        self.initial_balance = 1000.0
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 15.0
        self.leverage = 1  # No leverage
        self.balance_percentage = 90  # Use 90% of balance
        
        # Sample new listing symbols (you can modify this list)
        self.sample_symbols = [
            'NEWTOKENUSDT', 'LISTINGUSDT', 'FRESHUSDT', 'NEWCOINUSDT', 
            'LAUNCHUSDT', 'DEBUTUSDT', 'STARTUSDT', 'BEGINUSDT',
            'INITUSDT', 'FIRSTUSDT', 'PRIMEUSDT', 'ALPHATEST'
        ]
        
        self.results = []
    
    def generate_realistic_price_data(self, symbol: str, initial_price: float = None) -> pd.DataFrame:
        """Generate realistic 24-hour price data for a new listing"""
        if not initial_price:
            initial_price = random.uniform(0.01, 10.0)  # Random initial price
        
        # Generate 288 candles (24 hours * 12 five-minute intervals per hour)
        timestamps = []
        prices = []
        
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        current_price = initial_price
        
        # New listings often have high volatility patterns
        volatility_scenarios = [
            'pump_dump',     # Quick pump then dump
            'steady_decline', # Gradual decline
            'volatile_up',   # Volatile but trending up
            'volatile_down', # Volatile but trending down
            'sideways',      # Mostly sideways with some volatility
        ]
        
        scenario = random.choice(volatility_scenarios)
        
        for i in range(288):  # 24 hours of 5-minute candles
            timestamp = base_time + timedelta(minutes=i * 5)
            timestamps.append(timestamp)
            
            # Apply scenario-based price movement
            if scenario == 'pump_dump':
                if i < 36:  # First 3 hours - pump
                    change_pct = random.uniform(0.5, 3.0)
                elif i < 144:  # Next 9 hours - dump
                    change_pct = random.uniform(-2.0, -0.5)
                else:  # Rest - sideways with volatility
                    change_pct = random.uniform(-1.0, 1.0)
            
            elif scenario == 'steady_decline':
                change_pct = random.uniform(-1.5, 0.2)
            
            elif scenario == 'volatile_up':
                change_pct = random.uniform(-2.0, 2.5)
                if random.random() > 0.7:  # 30% chance of big move
                    change_pct *= 2
            
            elif scenario == 'volatile_down':
                change_pct = random.uniform(-2.5, 2.0)
                if random.random() > 0.7:  # 30% chance of big move
                    change_pct *= 2
            
            else:  # sideways
                change_pct = random.uniform(-1.0, 1.0)
            
            # Apply the change
            current_price *= (1 + change_pct / 100)
            current_price = max(current_price, 0.001)  # Prevent negative prices
            
            # Generate OHLC for this candle
            high = current_price * random.uniform(1.0, 1.02)
            low = current_price * random.uniform(0.98, 1.0)
            open_price = current_price * random.uniform(0.995, 1.005)
            close_price = current_price
            
            prices.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': random.uniform(10000, 100000)
            })
        
        df = pd.DataFrame(prices)
        df.set_index('timestamp', inplace=True)
        
        return df, scenario
    
    def simulate_trade(self, symbol: str, price_data: pd.DataFrame, scenario: str, 
                      current_balance: float) -> dict:
        """Simulate a single trade with SL and TP"""
        
        # Get entry price (very first price of the coin - opening price of first candle)
        entry_price = price_data.iloc[0]['open']
        
        # Calculate position size (no leverage)
        trade_balance = current_balance * (self.balance_percentage / 100)
        quantity = trade_balance / entry_price
        position_value = trade_balance
        
        # Calculate SL and TP prices
        stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + self.take_profit_pct / 100)
        
        # Initialize trade tracking
        trade_result = {
            'symbol': symbol,
            'scenario': scenario,
            'entry_time': price_data.index[0],
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'quantity': quantity,
            'position_value': position_value,
            'sl_hit_time': None,
            'tp_hit_time': None,
            'exit_time': None,
            'exit_price': None,
            'exit_reason': None,
            'pnl': 0,
            'pnl_pct': 0,
            'highest_price_24h': entry_price,
            'lowest_price_24h': entry_price,
            'highest_gain_pct': 0,
            'lowest_gain_pct': 0,
            'time_to_exit_minutes': 0,
            'max_drawdown_pct': 0,
            'max_profit_pct': 0
        }
        
        # Analyze each candle
        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Update highest and lowest prices
            if high_price > trade_result['highest_price_24h']:
                trade_result['highest_price_24h'] = high_price
            if low_price < trade_result['lowest_price_24h']:
                trade_result['lowest_price_24h'] = low_price
            
            # Calculate gains
            high_gain_pct = ((high_price - entry_price) / entry_price) * 100
            low_gain_pct = ((low_price - entry_price) / entry_price) * 100
            
            if high_gain_pct > trade_result['highest_gain_pct']:
                trade_result['highest_gain_pct'] = high_gain_pct
            if low_gain_pct < trade_result['lowest_gain_pct']:
                trade_result['lowest_gain_pct'] = low_gain_pct
            
            # Update max profit and drawdown
            current_pnl_pct = ((close_price - entry_price) / entry_price) * 100
            if current_pnl_pct > trade_result['max_profit_pct']:
                trade_result['max_profit_pct'] = current_pnl_pct
            if current_pnl_pct < trade_result['max_drawdown_pct']:
                trade_result['max_drawdown_pct'] = current_pnl_pct
            
            # Check for SL hit (check low first)
            if low_price <= stop_loss_price and not trade_result['sl_hit_time']:
                trade_result['sl_hit_time'] = timestamp
                trade_result['exit_time'] = timestamp
                trade_result['exit_price'] = stop_loss_price
                trade_result['exit_reason'] = 'Stop Loss'
                trade_result['time_to_exit_minutes'] = i * 5
                break
            
            # Check for TP hit (check high)
            if high_price >= take_profit_price and not trade_result['tp_hit_time']:
                trade_result['tp_hit_time'] = timestamp
                trade_result['exit_time'] = timestamp
                trade_result['exit_price'] = take_profit_price
                trade_result['exit_reason'] = 'Take Profit'
                trade_result['time_to_exit_minutes'] = i * 5
                break
        
        # If no SL or TP hit, exit at end of 24h
        if not trade_result['exit_time']:
            final_price = price_data.iloc[-1]['close']
            trade_result['exit_time'] = price_data.index[-1]
            trade_result['exit_price'] = final_price
            trade_result['exit_reason'] = '24h Timeout'
            trade_result['time_to_exit_minutes'] = 24 * 60
        
        # Calculate final PnL (no leverage)
        price_change = trade_result['exit_price'] - entry_price
        trade_result['pnl'] = (price_change / entry_price) * position_value
        trade_result['pnl_pct'] = (price_change / entry_price) * 100
        
        return trade_result
    
    def run_backtest(self, num_trades: int = 50):
        """Run the backtesting simulation"""
        print("🚀 Starting Manual Backtesting Simulation")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"📊 Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
        print(f"⚡ Leverage: {self.leverage}x (No leverage)")
        print(f"🎯 Number of Trades: {num_trades}")
        print("-" * 60)
        
        current_balance = self.initial_balance
        
        for i in range(num_trades):
            # Generate random symbol name
            symbol = f"COIN{i+1:03d}USDT"
            
            print(f"\n[{i+1}/{num_trades}] Simulating {symbol}...")
            
            # Generate price data
            price_data, scenario = self.generate_realistic_price_data(symbol)
            
            # Simulate trade
            trade_result = self.simulate_trade(symbol, price_data, scenario, current_balance)
            
            # Update balance
            current_balance += trade_result['pnl']
            trade_result['balance_after'] = current_balance
            trade_result['cumulative_return_pct'] = ((current_balance - self.initial_balance) / self.initial_balance) * 100
            
            # Store result
            self.results.append(trade_result)
            
            # Print trade summary
            print(f"✅ {symbol}: {trade_result['exit_reason']} | "
                  f"Scenario: {scenario} | "
                  f"PnL: ${trade_result['pnl']:.2f} ({trade_result['pnl_pct']:.2f}%) | "
                  f"Time: {trade_result['time_to_exit_minutes']} min | "
                  f"Balance: ${current_balance:.2f}")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze and print results"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        final_balance = df['balance_after'].iloc[-1]
        total_return = final_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Total Return: ${total_return:,.2f}")
        print(f"Total Return %: {total_return_pct:.2f}%")
        
        # Win/Loss Analysis
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        print(f"\n📈 TRADE ANALYSIS:")
        print(f"Total Trades: {len(df)}")
        print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(df)*100:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(df)*100:.1f}%)")
        
        if len(winning_trades) > 0:
            print(f"Average Win: ${winning_trades['pnl'].mean():.2f}")
            print(f"Largest Win: ${winning_trades['pnl'].max():.2f}")
        
        if len(losing_trades) > 0:
            print(f"Average Loss: ${losing_trades['pnl'].mean():.2f}")
            print(f"Largest Loss: ${losing_trades['pnl'].min():.2f}")
        
        print(f"Average Time to Exit: {df['time_to_exit_minutes'].mean():.1f} minutes")
        
        # Exit Reasons
        print(f"\n🎯 EXIT REASONS:")
        exit_reasons = df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"{reason}: {count} ({count/len(df)*100:.1f}%)")
        
        # Scenario Analysis
        print(f"\n📊 SCENARIO PERFORMANCE:")
        scenario_perf = df.groupby('scenario')['pnl'].agg(['count', 'mean', 'sum']).round(2)
        for scenario, stats in scenario_perf.iterrows():
            print(f"{scenario}: {stats['count']} trades, Avg: ${stats['mean']:.2f}, Total: ${stats['sum']:.2f}")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None):
        """Save results to CSV file"""
        if df.empty:
            print("❌ No results to save")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_backtest_results_{timestamp}.csv"
        
        # Prepare CSV data
        csv_data = []
        for _, row in df.iterrows():
            csv_row = {
                'Symbol': row['symbol'],
                'Scenario': row['scenario'],
                'Entry Time': row['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Entry Price': f"{row['entry_price']:.6f}",
                'Exit Time': row['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Price': f"{row['exit_price']:.6f}",
                'Exit Reason': row['exit_reason'],
                'Time to Exit (Minutes)': row['time_to_exit_minutes'],
                'Stop Loss Hit Time': row['sl_hit_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['sl_hit_time']) else '',
                'Take Profit Hit Time': row['tp_hit_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['tp_hit_time']) else '',
                'PnL ($)': f"{row['pnl']:.2f}",
                'PnL (%)': f"{row['pnl_pct']:.2f}",
                'Highest Price 24h': f"{row['highest_price_24h']:.6f}",
                'Lowest Price 24h': f"{row['lowest_price_24h']:.6f}",
                'Highest Gain (%)': f"{row['highest_gain_pct']:.2f}",
                'Lowest Gain (%)': f"{row['lowest_gain_pct']:.2f}",
                'Max Profit (%)': f"{row['max_profit_pct']:.2f}",
                'Max Drawdown (%)': f"{row['max_drawdown_pct']:.2f}",
                'Balance After': f"{row['balance_after']:.2f}",
                'Cumulative Return (%)': f"{row['cumulative_return_pct']:.2f}"
            }
            csv_data.append(csv_row)
        
        # Save to CSV
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
        
        return filename

def main():
    """Main function"""
    print("=" * 80)
    print("🎯 MANUAL BACKTESTING FOR NEW LISTING STRATEGY")
    print("=" * 80)
    
    # Initialize backtester
    backtester = ManualBacktest()
    
    # Run backtest
    results_df = backtester.run_backtest(num_trades=30)  # Test with 30 trades
    
    if not results_df.empty:
        # Save results to CSV
        backtester.save_to_csv(results_df)
        
        print("\n🎉 Backtesting completed! Check the CSV file for detailed results.")
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    main() 