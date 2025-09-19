#!/usr/bin/env python3

import requests
import pandas as pd
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CoinLaunchDateFinder:
    def __init__(self):
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"
        self.launch_dates = {}
        
        # Load all futures contracts and get last 200 coins
        with open('utils/futures.json', 'r') as f:
            all_futures = json.load(f)
        
        # Get last 200 coins only
        self.futures_list = all_futures[-200:]
        print(f"Will find launch dates for last {len(self.futures_list)} futures contracts")
        print(f"From: {self.futures_list[0]} to: {self.futures_list[-1]}")
    
    
    def get_first_candle_from_2019(self, symbol):
        """
        Get the very first candle from 2019 onwards
        """
        try:
            # Start from January 1, 2019
            start_time = int(datetime(2019, 1, 1).timestamp() * 1000)
            
            # Request first candle from 2019
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': start_time,
                'limit': 1
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            klines_data = response.json()
            
            if klines_data and len(klines_data) > 0:
                first_candle = klines_data[0]
                launch_timestamp = first_candle[0]  # open_time
                launch_date = datetime.fromtimestamp(launch_timestamp / 1000)
                return launch_date
            
            return None
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def find_all_launch_dates(self, delay=0.1):
        """
        Find launch dates for all symbols from 2019 onwards
        """
        print(f"Finding launch dates for {len(self.futures_list)} symbols from 2019...")
        print("=" * 80)
        
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(self.futures_list, 1):
            print(f"[{i}/{len(self.futures_list)}] ", end="")
            
            try:
                launch_date = self.get_first_candle_from_2019(symbol)
                if launch_date:
                    print(f"{symbol}: ✓ {launch_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    self.launch_dates[symbol] = launch_date.strftime('%Y-%m-%d %H:%M:%S')
                    successful += 1
                else:
                    print(f"{symbol}: ✗ No data since 2019")
                    failed += 1
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"{symbol}: ✗ Error - {e}")
                failed += 1
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {successful} successful, {failed} failed")
        
        return self.launch_dates
    
    def save_launch_dates(self, filename="coin_launch_dates_last200.json"):
        """Save launch dates to JSON file"""
        if self.launch_dates:
            with open(filename, 'w') as f:
                json.dump(self.launch_dates, f, indent=2, sort_keys=True)
            print(f"\n✅ Launch dates saved to {filename}")
            return filename
        else:
            print("No launch dates to save")
            return None
    
    def analyze_launch_dates(self):
        """Analyze the launch date data"""
        if not self.launch_dates:
            print("No data to analyze")
            return
        
        print("\n" + "=" * 80)
        print("LAUNCH DATE ANALYSIS (LAST 200 COINS)")
        print("=" * 80)
        
        # Convert to DataFrame for analysis
        data = []
        for symbol, date_str in self.launch_dates.items():
            launch_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            data.append({
                'symbol': symbol,
                'launch_date': launch_date,
                'year': launch_date.year,
                'month': launch_date.strftime('%Y-%m')
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('launch_date')
        
        print(f"📊 STATISTICS:")
        print(f"   Total coins analyzed: {len(df)}")
        print(f"   Earliest launch: {df['launch_date'].min().strftime('%Y-%m-%d %H:%M:%S')} ({df.iloc[0]['symbol']})")
        print(f"   Latest launch: {df['launch_date'].max().strftime('%Y-%m-%d %H:%M:%S')} ({df.iloc[-1]['symbol']})")
        
        print(f"\n📅 LAUNCHES BY YEAR:")
        year_counts = df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"   {year}: {count} coins")
        
        print(f"\n📈 MOST RECENT LAUNCHES (Last 10):")
        recent = df.tail(10)[['symbol', 'launch_date']]
        for _, row in recent.iterrows():
            print(f"   {row['symbol']}: {row['launch_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🏛️  OLDEST LAUNCHES IN THIS SET (First 10):")
        oldest = df.head(10)[['symbol', 'launch_date']]
        for _, row in oldest.iterrows():
            print(f"   {row['symbol']}: {row['launch_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df
    
    def get_launch_date_for_symbol(self, symbol):
        """Get launch date for a specific symbol"""
        if symbol in self.launch_dates:
            return self.launch_dates[symbol]
        
        print(f"Finding launch date for {symbol}...")
        launch_date = self.get_first_candle_from_2019(symbol)
        if launch_date:
            date_str = launch_date.strftime('%Y-%m-%d %H:%M:%S')
            self.launch_dates[symbol] = date_str
            return date_str
        return None

def load_existing_launch_dates(filename="coin_launch_dates_last200.json"):
    """Load previously saved launch dates"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No existing file found: {filename}")
        return {}
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {}

def main():
    print("=== Binance Futures Coin Launch Date Finder ===")
    print("Finding first candlestick date for LAST 200 futures contracts")
    print("=" * 80)
    
    finder = CoinLaunchDateFinder()
    
    # Check if we have existing data
    existing_data = load_existing_launch_dates()
    if existing_data:
        print(f"Found existing data for {len(existing_data)} coins")
        use_existing = input("Use existing data and only fetch missing coins? (y/N): ").strip().lower()
        if use_existing == 'y':
            finder.launch_dates = existing_data
            
            # Find missing coins
            missing_coins = [coin for coin in finder.futures_list if coin not in existing_data]
            if missing_coins:
                print(f"Found {len(missing_coins)} missing coins. Fetching their launch dates...")
                
                for symbol in missing_coins:
                    launch_date = finder.get_first_candle_from_2019(symbol)
                    if launch_date:
                        finder.launch_dates[symbol] = launch_date.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"✓ {symbol}: {launch_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        print(f"✗ {symbol}: No data since 2019")
                    time.sleep(0.1)
            else:
                print("All coins already have launch dates!")
    
    if not finder.launch_dates:
        print("Starting fresh data collection from 2019...")
        
        # Find all launch dates using 2019 method
        finder.find_all_launch_dates(delay=0.1)
    
    # Save results
    if finder.launch_dates:
        finder.save_launch_dates()
        
        # Analyze results
        df = finder.analyze_launch_dates()
        
        # Ask if user wants to export to CSV
        export_csv = input(f"\nExport results to CSV? (y/N): ").strip().lower()
        if export_csv == 'y':
            if 'df' in locals():
                csv_filename = "coin_launch_dates_last200.csv"
                df.to_csv(csv_filename, index=False)
                print(f"✅ Results exported to {csv_filename}")
    
    print(f"\n🎉 Launch date discovery complete for last 200 coins!")

if __name__ == "__main__":
    main()
