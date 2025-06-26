#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance Futures New Listing Backtesting System
Analyzes performance of trading new listings with 2% SL and 15% TP
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
warnings.filterwarnings('ignore')

# Add parent directory to path to import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FuturesListingBacktest:
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize the backtesting system"""
        self.client = None
        if api_key and api_secret:
            try:
                self.client = Client(api_key, api_secret, tld='com')
                print("✅ Binance client initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize Binance client: {e}")
                print("Will use public API only")
        
        # Trading parameters
        self.initial_balance = 1000.0
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 15.0
        self.leverage = 1  # No leverage
        
        # Results storage
        self.results = []
        self.balance_history = []
        
    def get_all_futures_symbols(self) -> List[Dict]:
        """Get all futures symbols with their listing information"""
        try:
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            symbols_info = []
            for symbol in data['symbols']:
                if symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT'):
                    symbols_info.append({
                        'symbol': symbol['symbol'],
                        'onboardDate': symbol.get('onboardDate', None),
                        'deliveryDate': symbol.get('deliveryDate', None),
                        'baseAsset': symbol['baseAsset'],
                        'quoteAsset': symbol['quoteAsset']
                    })
            
            print(f"📊 Found {len(symbols_info)} USDT futures pairs")
            return symbols_info
            
        except Exception as e:
            print(f"❌ Error fetching futures symbols: {e}")
            return []
    
    def get_symbol_listing_date(self, symbol: str) -> Optional[datetime]:
        """Get the listing date for a symbol by finding first available kline"""
        try:
            # Try to get the earliest kline data
            earliest_klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                start_str="1 Jan, 2020",
                limit=1
            ) if self.client else None
            
            if earliest_klines and len(earliest_klines) > 0:
                timestamp = earliest_klines[0][0]
                listing_date = datetime.fromtimestamp(timestamp / 1000)
                return listing_date
            
            return None
            
        except Exception as e:
            print(f"⚠️ Could not get listing date for {symbol}: {e}")
            return None
    
    def get_24h_price_data(self, symbol: str, start_time: datetime) -> Optional[pd.DataFrame]:
        """Get 24 hours of 5-minute price data from listing time"""
        try:
            if not self.client:
                print(f"⚠️ No Binance client available for {symbol}")
                return None
            
            end_time = start_time + timedelta(hours=24)
            
            # Get klines data
            klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                start_str=int(start_time.timestamp() * 1000),
                end_str=int(end_time.timestamp() * 1000)
            )
            
            if not klines or len(klines) == 0:
                print(f"⚠️ No price data available for {symbol}")
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
            
            print(f"📈 Got {len(df)} candles for {symbol} from {start_time}")
            return df
            
        except Exception as e:
            print(f"❌ Error getting price data for {symbol}: {e}")
            return None
    
    def simulate_trade(self, symbol: str, price_data: pd.DataFrame, entry_time: datetime, 
                      current_balance: float) -> Dict:
        """Simulate a single trade with SL and TP"""
        try:
            if price_data.empty:
                return None
            
            # Get entry price (very first price of the coin - opening price of first candle)
            entry_price = price_data.iloc[0]['open']
            
            # Calculate position size (no leverage, use 90% of balance)
            trade_balance = current_balance * 0.90  # Use 90% of balance
            quantity = trade_balance / entry_price
            position_value = trade_balance
            
            # Calculate SL and TP prices
            stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
            take_profit_price = entry_price * (1 + self.take_profit_pct / 100)
            
            # Track trade progress
            trade_result = {
                'symbol': symbol,
                'entry_time': entry_time,
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
                
                # Check for SL hit (check low first as it's more conservative)
                if low_price <= stop_loss_price and not trade_result['sl_hit_time']:
                    trade_result['sl_hit_time'] = timestamp
                    trade_result['exit_time'] = timestamp
                    trade_result['exit_price'] = stop_loss_price
                    trade_result['exit_reason'] = 'Stop Loss'
                    trade_result['time_to_exit_minutes'] = i * 5  # 5-minute intervals
                    break
                
                # Check for TP hit (check high)
                if high_price >= take_profit_price and not trade_result['tp_hit_time']:
                    trade_result['tp_hit_time'] = timestamp
                    trade_result['exit_time'] = timestamp
                    trade_result['exit_price'] = take_profit_price
                    trade_result['exit_reason'] = 'Take Profit'
                    trade_result['time_to_exit_minutes'] = i * 5  # 5-minute intervals
                    break
            
            # If no SL or TP hit, exit at end of 24h period
            if not trade_result['exit_time']:
                final_price = price_data.iloc[-1]['close']
                trade_result['exit_time'] = price_data.index[-1]
                trade_result['exit_price'] = final_price
                trade_result['exit_reason'] = '24h Timeout'
                trade_result['time_to_exit_minutes'] = 24 * 60  # 24 hours
            
            # Calculate final PnL (no leverage)
            price_change = trade_result['exit_price'] - entry_price
            trade_result['pnl'] = (price_change / entry_price) * position_value
            trade_result['pnl_pct'] = (price_change / entry_price) * 100
            
            return trade_result
            
        except Exception as e:
            print(f"❌ Error simulating trade for {symbol}: {e}")
            return None
    
    def get_recent_listings(self, days_back: int = 90) -> List[str]:
        """Get symbols that were listed in the last N days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            all_symbols = self.get_all_futures_symbols()
            
            recent_listings = []
            
            for symbol_info in all_symbols:
                symbol = symbol_info['symbol']
                
                # Skip perpetual contracts that have been around forever
                if symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 
                             'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'EOSUSDT']:
                    continue
                
                # Try to get listing date
                listing_date = self.get_symbol_listing_date(symbol)
                
                if listing_date and listing_date >= cutoff_date:
                    recent_listings.append(symbol)
                    print(f"📅 {symbol}: Listed on {listing_date.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Add small delay to avoid rate limits
                time.sleep(0.1)
            
            print(f"🎯 Found {len(recent_listings)} recent listings")
            return recent_listings
            
        except Exception as e:
            print(f"❌ Error getting recent listings: {e}")
            return []
    
    def run_backtest(self, symbols: List[str] = None, days_back: int = 90) -> pd.DataFrame:
        """Run the complete backtesting analysis"""
        print("🚀 Starting Futures Listing Backtest")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"📊 Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
        print(f"⚡ Leverage: {self.leverage}x (No leverage)")
        print("-" * 60)
        
        # Get symbols to test
        if not symbols:
            symbols = self.get_recent_listings(days_back)
        
        if not symbols:
            print("❌ No symbols found for backtesting")
            return pd.DataFrame()
        
        current_balance = self.initial_balance
        successful_trades = 0
        failed_trades = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Analyzing {symbol}...")
            
            try:
                # Get listing date
                listing_date = self.get_symbol_listing_date(symbol)
                if not listing_date:
                    print(f"⚠️ Could not determine listing date for {symbol}")
                    failed_trades += 1
                    continue
                
                # Get 24h price data
                price_data = self.get_24h_price_data(symbol, listing_date)
                if price_data is None or price_data.empty:
                    print(f"⚠️ No price data available for {symbol}")
                    failed_trades += 1
                    continue
                
                # Simulate trade
                trade_result = self.simulate_trade(symbol, price_data, listing_date, current_balance)
                if not trade_result:
                    print(f"⚠️ Could not simulate trade for {symbol}")
                    failed_trades += 1
                    continue
                
                # Update balance
                current_balance += trade_result['pnl']
                trade_result['balance_after'] = current_balance
                trade_result['cumulative_return_pct'] = ((current_balance - self.initial_balance) / self.initial_balance) * 100
                
                # Store result
                self.results.append(trade_result)
                successful_trades += 1
                
                # Print trade summary
                print(f"✅ {symbol}: {trade_result['exit_reason']} | "
                      f"PnL: ${trade_result['pnl']:.2f} ({trade_result['pnl_pct']:.2f}%) | "
                      f"Time: {trade_result['time_to_exit_minutes']} min | "
                      f"Balance: ${current_balance:.2f}")
                
                # Add delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                failed_trades += 1
                continue
        
        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Symbols Analyzed: {len(symbols)}")
        print(f"Successful Trades: {successful_trades}")
        print(f"Failed Analyses: {failed_trades}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${current_balance:,.2f}")
        print(f"Total Return: ${current_balance - self.initial_balance:,.2f}")
        print(f"Total Return %: {((current_balance - self.initial_balance) / self.initial_balance) * 100:.2f}%")
        
        if self.results:
            df = pd.DataFrame(self.results)
            return df
        else:
            return pd.DataFrame()
    
    def save_results_to_csv(self, df: pd.DataFrame, filename: str = None):
        """Save backtest results to CSV file"""
        if df.empty:
            print("❌ No results to save")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"futures_listing_backtest_{timestamp}.csv"
        
        # Prepare CSV columns
        csv_data = []
        for _, row in df.iterrows():
            csv_row = {
                'Symbol': row['symbol'],
                'Entry Time': row['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['entry_time']) else '',
                'Entry Price': f"{row['entry_price']:.6f}",
                'Exit Time': row['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['exit_time']) else '',
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
        print(f"💾 Results saved to: {filename}")
        
        return filename

def main():
    """Main function to run the backtest"""
    print("=" * 80)
    print("🎯 BINANCE FUTURES NEW LISTING BACKTESTING SYSTEM")
    print("=" * 80)
    
    # Initialize backtester
    # You can add your API keys here for more comprehensive data
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    backtester = FuturesListingBacktest(api_key, api_secret)
    
    # Run backtest for last 90 days
    print("Starting backtest analysis...")
    results_df = backtester.run_backtest(days_back=90)
    
    if not results_df.empty:
        # Save results
        filename = backtester.save_results_to_csv(results_df)
        
        # Print additional statistics
        print("\n📈 DETAILED STATISTICS")
        print("-" * 40)
        
        win_trades = results_df[results_df['pnl'] > 0]
        loss_trades = results_df[results_df['pnl'] < 0]
        
        print(f"Win Rate: {len(win_trades)}/{len(results_df)} ({len(win_trades)/len(results_df)*100:.1f}%)")
        print(f"Average Win: ${win_trades['pnl'].mean():.2f}" if len(win_trades) > 0 else "Average Win: $0.00")
        print(f"Average Loss: ${loss_trades['pnl'].mean():.2f}" if len(loss_trades) > 0 else "Average Loss: $0.00")
        print(f"Largest Win: ${results_df['pnl'].max():.2f}")
        print(f"Largest Loss: ${results_df['pnl'].min():.2f}")
        print(f"Average Time to Exit: {results_df['time_to_exit_minutes'].mean():.1f} minutes")
        
        # Exit reasons breakdown
        print(f"\n🎯 EXIT REASONS:")
        exit_reasons = results_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"{reason}: {count} ({count/len(results_df)*100:.1f}%)")
    
    else:
        print("❌ No results generated. Check API access and symbol availability.")

if __name__ == "__main__":
    main() 