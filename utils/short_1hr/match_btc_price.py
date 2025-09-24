#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BTCPriceAnalyzer:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        
    def get_btc_price_change(self, launch_date_str, timeframe_hours=24):
        """
        Get BTC price change for the specified timeframe starting from launch_date
        
        Args:
            launch_date_str: Launch date string (e.g., "2024-01-15 10:30:00")
            timeframe_hours: Hours to calculate price change (default: 24 hours)
        
        Returns:
            String: "+X.X%" or "-X.X%" representing BTC price change
        """
        try:
            # Parse launch date
            launch_datetime = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
            
            # Calculate start and end timestamps
            start_time = int(launch_datetime.timestamp() * 1000)
            end_time = int((launch_datetime + timedelta(hours=timeframe_hours)).timestamp() * 1000)
            
            # Get BTC price data
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',  # 1-hour intervals
                'startTime': start_time,
                'endTime': end_time,
                'limit': timeframe_hours + 1  # Get enough candles
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            klines_data = response.json()
            
            if len(klines_data) >= 2:
                # Get opening price (first candle) and closing price (last candle)
                open_price = float(klines_data[0][1])   # Open price of first candle
                close_price = float(klines_data[-1][4])  # Close price of last candle
                
                # Calculate percentage change
                price_change_pct = ((close_price - open_price) / open_price) * 100
                
                # Format as +X.X% or -X.X%
                if price_change_pct >= 0:
                    return f"+{price_change_pct:.2f}%"
                else:
                    return f"{price_change_pct:.2f}%"
            else:
                return "No Data"
                
        except Exception as e:
            print(f"Error getting BTC price for {launch_date_str}: {e}")
            return "Error"
    
    def process_backtest_file(self, input_file, output_file=None, timeframe_hours=24):
        """
        Process backtest results file and add BTC price change column
        
        Args:
            input_file: Path to the backtest results CSV file
            output_file: Path for output file (if None, overwrites input file)
            timeframe_hours: Hours to calculate BTC price change (default: 24 hours)
        """
        print(f"📊 === BTC Price Matcher ===")
        print(f"📁 Input file: {input_file}")
        print(f"⏰ Timeframe: {timeframe_hours} hours")
        print("=" * 60)
        
        try:
            # Read the backtest results file
            df = pd.read_csv(input_file)
            print(f"✅ Loaded {len(df)} trades from backtest file")
            
            # Check if launch_date column exists
            if 'launch_date' not in df.columns:
                print("❌ Error: 'launch_date' column not found in the file")
                print(f"Available columns: {list(df.columns)}")
                return
            
            # Initialize the new column
            df['btc_price_change'] = ""
            
            # Get unique launch dates to avoid duplicate API calls
            unique_dates = df['launch_date'].unique()
            print(f"🔍 Found {len(unique_dates)} unique launch dates")
            
            # Create a mapping of date to BTC price change
            btc_price_map = {}
            
            for i, launch_date in enumerate(unique_dates, 1):
                print(f"[{i:3d}/{len(unique_dates):3d}] Getting BTC price for {launch_date}...", end=" ")
                
                btc_change = self.get_btc_price_change(launch_date, timeframe_hours)
                btc_price_map[launch_date] = btc_change
                
                print(f"BTC: {btc_change}")
                
                # Rate limiting to avoid hitting API limits
                time.sleep(0.1)
            
            # Map BTC price changes to the dataframe
            df['btc_price_change'] = df['launch_date'].map(btc_price_map)
            
            # Determine output file
            if output_file is None:
                output_file = input_file
            
            # Save the updated file
            df.to_csv(output_file, index=False)
            
            print("\n" + "=" * 60)
            print("✅ BTC Price Analysis Complete!")
            print(f"💾 Updated file saved as: {output_file}")
            
            # Show summary statistics
            print(f"\n📊 BTC Price Change Summary:")
            valid_changes = df[df['btc_price_change'].str.contains('%', na=False)]['btc_price_change']
            
            if len(valid_changes) > 0:
                # Extract numerical values for statistics
                numeric_changes = []
                for change in valid_changes:
                    try:
                        num_val = float(change.replace('+', '').replace('%', ''))
                        numeric_changes.append(num_val)
                    except:
                        pass
                
                if numeric_changes:
                    positive_count = len([x for x in numeric_changes if x > 0])
                    negative_count = len([x for x in numeric_changes if x < 0])
                    zero_count = len([x for x in numeric_changes if x == 0])
                    
                    avg_change = np.mean(numeric_changes)
                    max_change = max(numeric_changes)
                    min_change = min(numeric_changes)
                    
                    print(f"   📈 Positive BTC days: {positive_count} ({positive_count/len(numeric_changes)*100:.1f}%)")
                    print(f"   📉 Negative BTC days: {negative_count} ({negative_count/len(numeric_changes)*100:.1f}%)")
                    print(f"   ➖ Neutral BTC days: {zero_count}")
                    print(f"   📊 Average BTC change: {avg_change:+.2f}%")
                    print(f"   🔝 Best BTC day: {max_change:+.2f}%")
                    print(f"   🔻 Worst BTC day: {min_change:+.2f}%")
            
            # Show sample of results
            print(f"\n📋 Sample Results (First 5 rows):")
            sample_cols = ['symbol', 'launch_date', 'profit_usd', 'exit_reason', 'btc_price_change']
            available_cols = [col for col in sample_cols if col in df.columns]
            print(df[available_cols].head().to_string(index=False))
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Error: File '{input_file}' not found")
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return None

def main():
    print("🚀 === BTC Price Matcher for Backtest Results ===")
    print("This script adds BTC price change data to your backtest results")
    print("=" * 70)
    
    # Configuration
    INPUT_FILE = "backtest_1m_results_20250918_015734.csv"
    TIMEFRAME_HOURS = 24  # Calculate BTC price change over 24 hours
    
    # You can modify these settings:
    print(f"📁 Default input file: {INPUT_FILE}")
    print(f"⏰ Default timeframe: {TIMEFRAME_HOURS} hours")
    print()
    
    # Ask user for input file (or use default)
    user_file = input(f"Enter CSV file path (or press Enter for default): ").strip()
    if user_file:
        INPUT_FILE = user_file
    
    # Ask for timeframe
    try:
        user_timeframe = input(f"Enter timeframe in hours (or press Enter for {TIMEFRAME_HOURS}h): ").strip()
        if user_timeframe:
            TIMEFRAME_HOURS = int(user_timeframe)
    except ValueError:
        print(f"Invalid timeframe, using default: {TIMEFRAME_HOURS} hours")
    
    print(f"\n🔍 Processing: {INPUT_FILE}")
    print(f"⏰ Timeframe: {TIMEFRAME_HOURS} hours")
    print("⚠️  This will fetch BTC price data from Binance API...")
    
    # Confirm before proceeding
    proceed = input("\nProceed? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Operation cancelled.")
        return
    
    # Initialize analyzer and process file
    analyzer = BTCPriceAnalyzer()
    result_df = analyzer.process_backtest_file(INPUT_FILE, timeframe_hours=TIMEFRAME_HOURS)
    
    if result_df is not None:
        print(f"\n🎉 Success! BTC price change data has been added to your file.")
        print(f"📊 You can now analyze correlation between your trade results and BTC price movements.")
    else:
        print(f"\n❌ Failed to process the file.")

if __name__ == "__main__":
    main()
