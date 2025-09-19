#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BTCPriceHistoricalAnalyzer:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        
    def get_btc_price_change_before_launch(self, launch_date_str, hours_before):
        """
        Get BTC price change for the specified period BEFORE launch_date
        
        Args:
            launch_date_str: Launch date string (e.g., "2024-01-15 10:30:00")
            hours_before: Hours before launch to analyze (e.g., 1, 2, 4, 8)
        
        Returns:
            String: "+X.X%" or "-X.X%" representing BTC price change in that period
        """
        try:
            # Parse launch date
            launch_datetime = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
            
            # Calculate start and end timestamps (going backwards from launch)
            end_time = int(launch_datetime.timestamp() * 1000)  # Launch time
            start_time = int((launch_datetime - timedelta(hours=hours_before)).timestamp() * 1000)  # X hours before
            
            # Get BTC price data for the period before launch
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',  # 1-hour intervals
                'startTime': start_time,
                'endTime': end_time,
                'limit': hours_before + 2  # Get enough candles + buffer
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            klines_data = response.json()
            
            if len(klines_data) >= 2:
                # Get opening price (start of period) and closing price (end of period/launch time)
                open_price = float(klines_data[0][1])    # Open price of first candle (X hours before)
                close_price = float(klines_data[-1][4])  # Close price of last candle (at launch time)
                
                # Calculate percentage change over the period
                price_change_pct = ((close_price - open_price) / open_price) * 100
                
                # Format as +X.X% or -X.X%
                if price_change_pct >= 0:
                    return f"+{price_change_pct:.2f}%"
                else:
                    return f"{price_change_pct:.2f}%"
            else:
                return "No Data"
                
        except Exception as e:
            print(f"Error getting BTC price for {launch_date_str} ({hours_before}h before): {e}")
            return "Error"
    
    def process_backtest_file(self, input_file, output_file=None):
        """
        Process backtest results file and add BTC historical price change columns
        
        Args:
            input_file: Path to the backtest results CSV file
            output_file: Path for output file (if None, overwrites input file)
        """
        print(f"📊 === BTC Historical Price Matcher ===")
        print(f"📁 Input file: {input_file}")
        print(f"⏰ Analyzing BTC price changes: 1h, 2h, 4h, 8h BEFORE launch")
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
            
            # Initialize the new columns
            time_periods = [1, 2, 4, 8]  # Hours before launch
            column_names = [f'btc_change_{h}h_before' for h in time_periods]
            
            for col in column_names:
                df[col] = ""
            
            # Get unique launch dates to avoid duplicate API calls
            unique_dates = df['launch_date'].unique()
            print(f"🔍 Found {len(unique_dates)} unique launch dates")
            
            # Create mappings for each time period
            btc_price_maps = {period: {} for period in time_periods}
            
            total_api_calls = len(unique_dates) * len(time_periods)
            current_call = 0
            
            for i, launch_date in enumerate(unique_dates, 1):
                print(f"[{i:3d}/{len(unique_dates):3d}] Analyzing {launch_date}:")
                
                for period in time_periods:
                    current_call += 1
                    print(f"    [{current_call:3d}/{total_api_calls:3d}] {period}h before...", end=" ")
                    
                    btc_change = self.get_btc_price_change_before_launch(launch_date, period)
                    btc_price_maps[period][launch_date] = btc_change
                    
                    print(f"BTC: {btc_change}")
                    
                    # Rate limiting to avoid hitting API limits
                    time.sleep(0.1)
            
            # Map BTC price changes to the dataframe
            for i, period in enumerate(time_periods):
                column_name = column_names[i]
                df[column_name] = df['launch_date'].map(btc_price_maps[period])
            
            # Determine output file
            if output_file is None:
                output_file = input_file.replace('.csv', '_with_btc_historical.csv')
            
            # Save the updated file
            df.to_csv(output_file, index=False)
            
            print("\n" + "=" * 60)
            print("✅ BTC Historical Price Analysis Complete!")
            print(f"💾 Updated file saved as: {output_file}")
            
            # Show summary statistics for each time period
            print(f"\n📊 BTC Historical Price Change Summary:")
            
            for i, period in enumerate(time_periods):
                column_name = column_names[i]
                valid_changes = df[df[column_name].str.contains('%', na=False)][column_name]
                
                if len(valid_changes) > 0:
                    print(f"\n🕐 {period} Hour(s) Before Launch:")
                    
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
                        
                        print(f"   📈 Positive: {positive_count} ({positive_count/len(numeric_changes)*100:.1f}%)")
                        print(f"   📉 Negative: {negative_count} ({negative_count/len(numeric_changes)*100:.1f}%)")
                        print(f"   📊 Average: {avg_change:+.2f}%")
                        print(f"   🔝 Best: {max_change:+.2f}%")
                        print(f"   🔻 Worst: {min_change:+.2f}%")
            
            # Show sample of results
            print(f"\n📋 Sample Results (First 3 rows):")
            sample_cols = ['symbol', 'launch_date', 'profit_usd', 'exit_reason'] + column_names
            available_cols = [col for col in sample_cols if col in df.columns]
            print(df[available_cols].head(3).to_string(index=False))
            
            # Analysis insights
            print(f"\n🔍 Analysis Insights:")
            print(f"📊 You can now analyze:")
            print(f"   • Does BTC momentum before launch affect coin performance?")
            print(f"   • Are launches timed with BTC trends?")
            print(f"   • Which time period (1h, 2h, 4h, 8h) shows strongest correlation?")
            print(f"   • Do coins perform better when BTC was rising before launch?")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Error: File '{input_file}' not found")
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return None

def main():
    print("🚀 === BTC Historical Price Matcher ===")
    print("This script adds BTC price change data BEFORE each coin launch")
    print("Analyzes: 1h, 2h, 4h, 8h periods before launch date")
    print("=" * 70)
    
    # Configuration
    INPUT_FILE = "backtest_1m_results_20250918_015734.csv"
    
    print(f"📁 Default input file: {INPUT_FILE}")
    print()
    
    # Ask user for input file (or use default)
    user_file = input(f"Enter CSV file path (or press Enter for default): ").strip()
    if user_file:
        INPUT_FILE = user_file
    
    print(f"\n🔍 Processing: {INPUT_FILE}")
    print("📈 This will add 4 new columns:")
    print("   • btc_change_1h_before")
    print("   • btc_change_2h_before") 
    print("   • btc_change_4h_before")
    print("   • btc_change_8h_before")
    print("\n⚠️  This will make multiple API calls to fetch BTC historical data...")
    
    # Show estimated API calls
    try:
        df_test = pd.read_csv(INPUT_FILE)
        unique_dates = df_test['launch_date'].nunique()
        total_calls = unique_dates * 4
        estimated_time = total_calls * 0.1 / 60  # 0.1s delay per call
        print(f"📊 Estimated API calls: {total_calls} ({unique_dates} dates × 4 periods)")
        print(f"⏱️  Estimated time: ~{estimated_time:.1f} minutes")
    except:
        print("📊 Unable to estimate API calls (file read error)")
    
    # Confirm before proceeding
    proceed = input("\nProceed? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Operation cancelled.")
        return
    
    # Initialize analyzer and process file
    analyzer = BTCPriceHistoricalAnalyzer()
    result_df = analyzer.process_backtest_file(INPUT_FILE)
    
    if result_df is not None:
        print(f"\n🎉 Success! BTC historical price data has been added.")
        print(f"📊 You can now analyze how BTC momentum before launch affects trading results.")
        print(f"💡 Look for patterns: Do coins perform better when BTC was rising in the hours before launch?")
    else:
        print(f"\n❌ Failed to process the file.")

if __name__ == "__main__":
    main()
