#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Binance Futures Backtesting System
Goes through ALL futures pairs and backtests from their listing date
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from binance.client import Client
import time
from typing import Dict, List, Tuple, Optional
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from parent directory or current directory
load_dotenv()  # Looks for .env in current directory
load_dotenv('../.env')  # Also try parent directory

class ComprehensiveBacktest:
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize the comprehensive backtesting system"""
        self.client = None
        if api_key and api_secret:
            try:
                self.client = Client(api_key, api_secret, tld='com')
                print("✅ Binance client initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize Binance client: {e}")
                print("❌ API credentials required for historical data")
                sys.exit(1)
        else:
            print("❌ API credentials required for historical data")
            sys.exit(1)
        
        # Trading parameters
        self.initial_balance = 1000.0
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 15.0
        self.leverage = 1  # No leverage
        self.balance_percentage = 90  # Use 90% of balance
        
        # Results storage
        self.results = []
        self.current_balance = self.initial_balance
        
    def get_all_futures_symbols(self) -> List[str]:
        """Get all USDT futures symbols"""
        try:
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            symbols = []
            for symbol in data['symbols']:
                if (symbol['status'] == 'TRADING' and 
                    symbol['symbol'].endswith('USDT') and
                    symbol['contractType'] == 'PERPETUAL'):
                    symbols.append(symbol['symbol'])
            
            print(f"📊 Found {len(symbols)} USDT perpetual futures pairs")
            return sorted(symbols)
            
        except Exception as e:
            print(f"❌ Error fetching futures symbols: {e}")
            return []
    
    def get_symbol_first_listing_date(self, symbol: str) -> Optional[datetime]:
        """Get the earliest available data for a symbol"""
        try:
            # Try to get the earliest kline data
            earliest_klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                start_str="1 Jan, 2019",  # Start from early date
                limit=1
            )
            
            if earliest_klines and len(earliest_klines) > 0:
                timestamp = earliest_klines[0][0]
                listing_date = datetime.fromtimestamp(timestamp / 1000)
                return listing_date
            
            return None
            
        except Exception as e:
            print(f"⚠️ Could not get listing date for {symbol}: {e}")
            return None
    
    def get_first_24h_data(self, symbol: str, start_time: datetime) -> Optional[pd.DataFrame]:
        """Get first 24 hours of 5-minute data from listing"""
        try:
            end_time = start_time + timedelta(hours=24)
            
            # Get klines data for first 24 hours
            klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                start_str=int(start_time.timestamp() * 1000),
                end_str=int(end_time.timestamp() * 1000)
            )
            
            if not klines or len(klines) < 10:  # Need at least some data
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting price data for {symbol}: {e}")
            return None
    
    def simulate_trade(self, symbol: str, price_data: pd.DataFrame, listing_date: datetime) -> Optional[Dict]:
        """Simulate a single trade with SL and TP"""
        try:
            if price_data.empty:
                return None
            
            # Get entry price (very first price of the coin)
            entry_price = price_data.iloc[0]['open']
            
            # Calculate position size (no leverage, use 90% of balance)
            trade_balance = self.current_balance * (self.balance_percentage / 100)
            quantity = trade_balance / entry_price
            position_value = trade_balance
            
            # Calculate SL and TP prices
            stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
            take_profit_price = entry_price * (1 + self.take_profit_pct / 100)
            
            # Track trade progress
            trade_result = {
                'symbol': symbol,
                'listing_date': listing_date,
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
                trade_result['time_to_exit_minutes'] = len(price_data) * 5
            
            # Calculate final PnL (no leverage)
            price_change = trade_result['exit_price'] - entry_price
            trade_result['pnl'] = (price_change / entry_price) * position_value
            trade_result['pnl_pct'] = (price_change / entry_price) * 100
            
            return trade_result
            
        except Exception as e:
            print(f"❌ Error simulating trade for {symbol}: {e}")
            return None
    
    def run_comprehensive_backtest(self) -> pd.DataFrame:
        """Run backtest on all futures pairs"""
        print("🚀 Starting Comprehensive Futures Backtesting")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"📊 Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
        print(f"⚡ Leverage: {self.leverage}x (No leverage)")
        print("=" * 80)
        
        # Get all futures symbols
        symbols = self.get_all_futures_symbols()
        if not symbols:
            print("❌ No symbols found")
            return pd.DataFrame()
        
        print(f"🎯 Processing {len(symbols)} futures pairs...")
        print("=" * 80)
        
        successful_trades = 0
        failed_analyses = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i:3d}/{len(symbols)}] 🔍 Analyzing {symbol}...")
            
            try:
                # Get listing date
                listing_date = self.get_symbol_first_listing_date(symbol)
                if not listing_date:
                    print(f"    ❌ Could not get listing date")
                    failed_analyses += 1
                    continue
                
                print(f"    📅 Listed: {listing_date.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get first 24h price data
                price_data = self.get_first_24h_data(symbol, listing_date)
                if price_data is None or price_data.empty:
                    print(f"    ❌ No price data available")
                    failed_analyses += 1
                    continue
                
                print(f"    📈 Got {len(price_data)} candles")
                
                # Simulate trade
                trade_result = self.simulate_trade(symbol, price_data, listing_date)
                if not trade_result:
                    print(f"    ❌ Could not simulate trade")
                    failed_analyses += 1
                    continue
                
                # Update balance
                self.current_balance += trade_result['pnl']
                trade_result['balance_after'] = self.current_balance
                trade_result['cumulative_return_pct'] = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
                
                # Store result
                self.results.append(trade_result)
                successful_trades += 1
                
                # Print trade summary
                status_emoji = "✅" if trade_result['pnl'] > 0 else "❌"
                print(f"    {status_emoji} {trade_result['exit_reason']} | "
                      f"Entry: ${trade_result['entry_price']:.6f} | "
                      f"Exit: ${trade_result['exit_price']:.6f}")
                print(f"    💰 PnL: ${trade_result['pnl']:.2f} ({trade_result['pnl_pct']:.2f}%) | "
                      f"Time: {trade_result['time_to_exit_minutes']} min")
                print(f"    📊 Balance: ${self.current_balance:.2f} | "
                      f"Total Return: {trade_result['cumulative_return_pct']:.2f}%")
                
                # Add delay to avoid rate limits
                time.sleep(0.1)
                
                # Save progress every 50 trades
                if successful_trades % 50 == 0:
                    self.save_progress_csv(successful_trades)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                failed_analyses += 1
                continue
        
        # Final summary
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BACKTEST SUMMARY")
        print("="*80)
        print(f"Total Symbols Processed: {len(symbols)}")
        print(f"Successful Trades: {successful_trades}")
        print(f"Failed Analyses: {failed_analyses}")
        print(f"Success Rate: {successful_trades/len(symbols)*100:.1f}%")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: ${self.current_balance - self.initial_balance:,.2f}")
        print(f"Total Return %: {((self.current_balance - self.initial_balance) / self.initial_balance) * 100:.2f}%")
        
        if self.results:
            df = pd.DataFrame(self.results)
            self.print_detailed_stats(df)
            return df
        else:
            return pd.DataFrame()
    
    def print_detailed_stats(self, df: pd.DataFrame):
        """Print detailed statistics"""
        if df.empty:
            return
        
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        print(f"\n📈 DETAILED STATISTICS")
        print("-" * 50)
        print(f"Win Rate: {len(winning_trades)}/{len(df)} ({len(winning_trades)/len(df)*100:.1f}%)")
        
        if len(winning_trades) > 0:
            print(f"Average Win: ${winning_trades['pnl'].mean():.2f}")
            print(f"Largest Win: ${winning_trades['pnl'].max():.2f}")
            print(f"Median Win: ${winning_trades['pnl'].median():.2f}")
        
        if len(losing_trades) > 0:
            print(f"Average Loss: ${losing_trades['pnl'].mean():.2f}")
            print(f"Largest Loss: ${losing_trades['pnl'].min():.2f}")
            print(f"Median Loss: ${losing_trades['pnl'].median():.2f}")
        
        print(f"Average Time to Exit: {df['time_to_exit_minutes'].mean():.1f} minutes")
        print(f"Median Time to Exit: {df['time_to_exit_minutes'].median():.1f} minutes")
        
        # Exit reasons breakdown
        print(f"\n🎯 EXIT REASONS:")
        exit_reasons = df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"{reason}: {count} ({count/len(df)*100:.1f}%)")
        
        # Performance by year
        df['listing_year'] = pd.to_datetime(df['listing_date']).dt.year
        yearly_perf = df.groupby('listing_year')['pnl'].agg(['count', 'sum', 'mean']).round(2)
        print(f"\n📅 PERFORMANCE BY LISTING YEAR:")
        for year, stats in yearly_perf.iterrows():
            print(f"{year}: {stats['count']} trades, Total: ${stats['sum']:.2f}, Avg: ${stats['mean']:.2f}")
    
    def save_progress_csv(self, trade_count: int):
        """Save progress to CSV"""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_progress_{trade_count}trades_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        self.save_results_to_csv(df, filename)
        print(f"    💾 Progress saved: {filename}")
    
    def save_results_to_csv(self, df: pd.DataFrame, filename: str = None):
        """Save backtest results to CSV file"""
        if df.empty:
            print("❌ No results to save")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_backtest_results_{timestamp}.csv"
        
        # Prepare CSV columns
        csv_data = []
        for _, row in df.iterrows():
            csv_row = {
                'Symbol': row['symbol'],
                'Listing Date': row['listing_date'].strftime('%Y-%m-%d %H:%M:%S'),
                'Entry Time': row['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Entry Price': f"{row['entry_price']:.8f}",
                'Exit Time': row['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Price': f"{row['exit_price']:.8f}",
                'Exit Reason': row['exit_reason'],
                'Time to Exit (Minutes)': row['time_to_exit_minutes'],
                'Stop Loss Hit Time': row['sl_hit_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['sl_hit_time']) else '',
                'Take Profit Hit Time': row['tp_hit_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['tp_hit_time']) else '',
                'PnL ($)': f"{row['pnl']:.2f}",
                'PnL (%)': f"{row['pnl_pct']:.2f}",
                'Highest Price 24h': f"{row['highest_price_24h']:.8f}",
                'Lowest Price 24h': f"{row['lowest_price_24h']:.8f}",
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
        print(f"💾 Results saved to: {filename}")
        
        return filename

def main():
    """Main function to run the comprehensive backtest"""
    print("=" * 80)
    print("🎯 COMPREHENSIVE BINANCE FUTURES BACKTESTING SYSTEM")
    print("🔍 Analyzing ALL futures pairs from their listing dates")
    print("=" * 80)
    
    # Get API credentials from .env file
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Please set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file")
        print("Example .env file:")
        print("BINANCE_API_KEY=your_api_key_here")
        print("BINANCE_API_SECRET=your_api_secret_here")
        print("\nMake sure the .env file is in the same directory as this script")
        return
    
    # Initialize backtester
    backtester = ComprehensiveBacktest(api_key, api_secret)
    
    # Run comprehensive backtest
    print("Starting comprehensive analysis of all futures pairs...")
    results_df = backtester.run_comprehensive_backtest()
    
    if not results_df.empty:
        # Save final results
        filename = backtester.save_results_to_csv(results_df)
        print(f"\n🎉 Comprehensive backtesting completed!")
        print(f"📁 Final results saved to: {filename}")
        print(f"📊 Total trades analyzed: {len(results_df)}")
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    main() 