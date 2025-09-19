#!/usr/bin/env python3

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LaunchDayDataDownloader:
    def __init__(self):
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"
        self.data_folder = "launch_day_data"
        self.launch_dates = {}
        
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Load launch dates
        self.load_launch_dates()
        
    def load_launch_dates(self):
        """Load launch dates from the previously created file"""
        launch_files = [
            "coin_launch_dates_last200.json",
            "utils/1_hr/coin_launch_dates_last200.json",
            "coin_launch_dates.json"
        ]
        
        for filename in launch_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        self.launch_dates = json.load(f)
                    print(f"✅ Loaded launch dates from {filename}")
                    print(f"Found {len(self.launch_dates)} coins with launch dates")
                    return
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
        
        print("❌ No launch dates file found! Please run coin_launch_dates.py first.")
        exit(1)
    
    def download_24h_from_launch(self, symbol, launch_date_str):
        """
        Download exactly 24 candles (24 hours) from launch date
        """
        try:
            # Parse launch date
            launch_date = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
            start_time = int(launch_date.timestamp() * 1000)
            
            print(f"Downloading 24h data for {symbol} from {launch_date_str}...", end=" ")
            
            # Request exactly 24 candles from launch time
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': start_time,
                'limit': 24
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            klines_data = response.json()
            
            if not klines_data or len(klines_data) == 0:
                print("❌ No data available")
                return None
            
            print(f"✅ Got {len(klines_data)} candles")
            return klines_data
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def process_klines_data(self, klines_data, symbol, launch_date_str):
        """Convert raw klines data to pandas DataFrame"""
        if not klines_data:
            return None
        
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines_data, columns=columns)
        
        # Convert to appropriate data types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only essential columns for backtesting
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        
        # Add metadata
        df['symbol'] = symbol
        df['launch_date'] = launch_date_str
        df['hours_from_launch'] = range(len(df))
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def save_data(self, symbol, df, launch_date_str):
        """Save DataFrame to CSV file"""
        # Create filename with launch date for clarity
        launch_date_clean = launch_date_str.replace(' ', '_').replace(':', '-')
        filename = f"{self.data_folder}/{symbol}_launch_{launch_date_clean}.csv"
        
        df.to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")
        return filename
    
    def download_all_launch_day_data(self, delay=0.1):
        """Download 24h data from launch for all coins"""
        print(f"🚀 Starting download of 24h launch data for {len(self.launch_dates)} coins...")
        print(f"Data will be saved to '{self.data_folder}/' folder")
        print("=" * 80)
        
        successful_downloads = []
        failed_downloads = []
        
        for i, (symbol, launch_date_str) in enumerate(self.launch_dates.items(), 1):
            print(f"[{i}/{len(self.launch_dates)}] ", end="")
            
            try:
                # Check if file already exists
                launch_date_clean = launch_date_str.replace(' ', '_').replace(':', '-')
                filename = f"{self.data_folder}/{symbol}_launch_{launch_date_clean}.csv"
                
                if os.path.exists(filename):
                    existing_df = pd.read_csv(filename)
                    print(f"{symbol}: ✅ Already exists ({len(existing_df)} candles)")
                    successful_downloads.append(symbol)
                    continue
                
                # Download 24h data from launch
                klines_data = self.download_24h_from_launch(symbol, launch_date_str)
                
                if klines_data:
                    # Process and save
                    df = self.process_klines_data(klines_data, symbol, launch_date_str)
                    if df is not None and len(df) > 0:
                        self.save_data(symbol, df, launch_date_str)
                        successful_downloads.append(symbol)
                    else:
                        print(f"{symbol}: ❌ Failed to process data")
                        failed_downloads.append(symbol)
                else:
                    failed_downloads.append(symbol)
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"{symbol}: ❌ Error - {e}")
                failed_downloads.append(symbol)
        
        print("\n" + "=" * 80)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {len(successful_downloads)}")
        print(f"❌ Failed: {len(failed_downloads)}")
        
        if failed_downloads:
            print(f"\n❌ Failed symbols: {failed_downloads}")
        
        return successful_downloads, failed_downloads
    
    def analyze_launch_data(self):
        """Analyze the downloaded launch day data"""
        print("\n🔍 ANALYZING LAUNCH DAY DATA...")
        print("=" * 80)
        
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        
        if not csv_files:
            print("No data files found to analyze")
            return
        
        analysis_data = []
        
        for filename in csv_files:
            try:
                filepath = os.path.join(self.data_folder, filename)
                df = pd.read_csv(filepath)
                
                if len(df) > 0:
                    symbol = df['symbol'].iloc[0]
                    launch_date = df['launch_date'].iloc[0]
                    
                    # Calculate first 24h performance
                    first_price = df['open'].iloc[0]
                    last_price = df['close'].iloc[-1]
                    high_24h = df['high'].max()
                    low_24h = df['low'].min()
                    
                    performance_24h = ((last_price - first_price) / first_price) * 100
                    max_gain_24h = ((high_24h - first_price) / first_price) * 100
                    max_loss_24h = ((low_24h - first_price) / first_price) * 100
                    
                    total_volume = df['volume'].sum()
                    
                    analysis_data.append({
                        'symbol': symbol,
                        'launch_date': launch_date,
                        'first_price': first_price,
                        'last_price_24h': last_price,
                        'performance_24h': performance_24h,
                        'max_gain_24h': max_gain_24h,
                        'max_loss_24h': max_loss_24h,
                        'high_24h': high_24h,
                        'low_24h': low_24h,
                        'total_volume_24h': total_volume,
                        'candles_count': len(df)
                    })
                    
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
        
        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            
            print(f"📈 LAUNCH DAY PERFORMANCE ANALYSIS:")
            print(f"   Total coins analyzed: {len(analysis_df)}")
            print(f"   Average 24h performance: {analysis_df['performance_24h'].mean():.2f}%")
            print(f"   Best 24h performance: {analysis_df['performance_24h'].max():.2f}% ({analysis_df.loc[analysis_df['performance_24h'].idxmax(), 'symbol']})")
            print(f"   Worst 24h performance: {analysis_df['performance_24h'].min():.2f}% ({analysis_df.loc[analysis_df['performance_24h'].idxmin(), 'symbol']})")
            
            print(f"\n🏆 TOP 10 BEST LAUNCH DAY PERFORMERS:")
            top_performers = analysis_df.nlargest(10, 'performance_24h')[['symbol', 'performance_24h', 'max_gain_24h', 'launch_date']]
            print(top_performers.to_string(index=False))
            
            print(f"\n📉 TOP 10 WORST LAUNCH DAY PERFORMERS:")
            worst_performers = analysis_df.nsmallest(10, 'performance_24h')[['symbol', 'performance_24h', 'max_loss_24h', 'launch_date']]
            print(worst_performers.to_string(index=False))
            
            # Save analysis
            analysis_filename = f"{self.data_folder}/launch_day_analysis.csv"
            analysis_df.to_csv(analysis_filename, index=False)
            print(f"\n💾 Analysis saved to {analysis_filename}")
            
            return analysis_df
        
        return None
    
    def show_sample_data(self, num_samples=3):
        """Show sample data for a few coins"""
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv') and 'analysis' not in f]
        
        if not csv_files:
            print("No sample data to show")
            return
        
        print(f"\n📋 SAMPLE DATA (First {num_samples} coins):")
        print("=" * 80)
        
        for filename in csv_files[:num_samples]:
            try:
                filepath = os.path.join(self.data_folder, filename)
                df = pd.read_csv(filepath)
                
                symbol = df['symbol'].iloc[0]
                launch_date = df['launch_date'].iloc[0]
                
                print(f"\n🪙 {symbol} - Launch: {launch_date}")
                print(f"First 5 hours of trading:")
                sample_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'hours_from_launch']].head()
                print(sample_df.to_string(index=False))
                
            except Exception as e:
                print(f"Error showing sample for {filename}: {e}")

def main():
    print("🚀 === Launch Day Data Downloader ===")
    print("Downloads 24 hours of 1h candlestick data from each coin's launch date")
    print("=" * 80)
    
    downloader = LaunchDayDataDownloader()
    
    if not downloader.launch_dates:
        print("❌ No launch dates available. Please run coin_launch_dates.py first.")
        return
    
    print(f"📋 Will download 24h launch data for {len(downloader.launch_dates)} coins")
    
    proceed = input(f"\n🤔 Proceed with download? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Download cancelled.")
        return
    
    # Download all launch day data
    successful, failed = downloader.download_all_launch_day_data(delay=0.1)
    
    if successful:
        # Show sample data
        downloader.show_sample_data(3)
        
        # Analyze launch day performance
        analysis_df = downloader.analyze_launch_data()
        
        print(f"\n✅ Launch day data download complete!")
        print(f"📁 Data saved in: {os.path.abspath(downloader.data_folder)}/")
        print(f"🎯 Ready for launch day analysis with {len(successful)} coins")
    else:
        print("❌ No successful downloads")

if __name__ == "__main__":
    main()
