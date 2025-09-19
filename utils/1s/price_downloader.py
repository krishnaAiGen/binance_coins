#!/usr/bin/env python3

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OneMinuteDataDownloader:
    def __init__(self):
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"
        self.data_folder = "1m_data"
        self.launch_dates = {}
        
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Load launch dates
        self.load_launch_dates()
        
    def load_launch_dates(self):
        """Load launch dates from the previously created file"""
        launch_files = [
            "/Users/krishnayadav/Documents/test_projects/binance_coins/utils/1_hr/coin_launch_dates_last200.json",
            "/Users/krishnayadav/Documents/test_projects/binance_coins/utils/1_hr/coin_launch_dates_last200.csv",
            "utils/1_hr/coin_launch_dates_last200.json",
            "utils/1_hr/coin_launch_dates_last200.csv",
            "coin_launch_dates_last200.json",
            "coin_launch_dates_last200.csv",
            "coin_launch_dates.json"
        ]
        
        for filename in launch_files:
            if os.path.exists(filename):
                try:
                    if filename.endswith('.json'):
                        with open(filename, 'r') as f:
                            self.launch_dates = json.load(f)
                    elif filename.endswith('.csv'):
                        # Load CSV and convert to dictionary
                        df = pd.read_csv(filename)
                        if 'symbol' in df.columns and 'launch_date' in df.columns:
                            self.launch_dates = dict(zip(df['symbol'], df['launch_date']))
                        else:
                            print(f"❌ CSV file {filename} missing required columns (symbol, launch_date)")
                            continue
                    
                    print(f"✅ Loaded launch dates from {filename}")
                    print(f"Found {len(self.launch_dates)} coins with launch dates")
                    return
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
        
        print("❌ No launch dates file found! Please run coin_launch_dates.py first.")
        exit(1)
    
    def download_4h_minutes_from_launch(self, symbol, launch_date_str, batch_size=240):
        """
        Download 4 hours of 1-minute candles (240 candles) from launch date
        Single API call since 240 < 1000 limit
        """
        try:
            # Parse launch date
            launch_date = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
            start_time = int(launch_date.timestamp() * 1000)
            
            target_candles = 240  # 4 hours * 60 minutes
            
            print(f"Downloading {target_candles} 1m candles for {symbol}...", end=" ")
            
            # Single API call for 240 minutes (within Binance's 1000 limit)
            params = {
                'symbol': symbol,
                'interval': '1m',
                'startTime': start_time,
                'limit': target_candles
            }
            
            response = requests.get(self.base_url, params=params, timeout=20)
            response.raise_for_status()
            klines_data = response.json()
            
            if not klines_data or len(klines_data) == 0:
                print("❌ No data available")
                return None
            
            print(f"✅ {len(klines_data)} candles")
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
        df['minutes_from_launch'] = range(len(df))
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def save_data(self, symbol, df, launch_date_str):
        """Save DataFrame to CSV file"""
        # Create filename with launch date for clarity
        launch_date_clean = launch_date_str.replace(' ', '_').replace(':', '-')
        filename = f"{self.data_folder}/{symbol}_1m_launch_{launch_date_clean}.csv"
        
        df.to_csv(filename, index=False)
        print(f"💾 Saved to {filename} ({len(df)} records)")
        return filename
    
    def filter_valid_launch_dates(self):
        """Filter out future dates and invalid dates"""
        current_time = datetime.now()
        valid_dates = {}
        future_dates = []
        invalid_dates = []
        
        for symbol, launch_date_str in self.launch_dates.items():
            try:
                launch_date = datetime.strptime(launch_date_str, '%Y-%m-%d %H:%M:%S')
                
                # Check if launch date is in the past (at least 1 day ago for safety)
                if launch_date < current_time - timedelta(days=1):
                    valid_dates[symbol] = launch_date_str
                else:
                    future_dates.append((symbol, launch_date_str))
            except:
                invalid_dates.append((symbol, launch_date_str))
        
        print(f"📊 Date filtering results:")
        print(f"   ✅ Valid dates (past): {len(valid_dates)}")
        print(f"   ⏳ Future dates (skipped): {len(future_dates)}")
        print(f"   ❌ Invalid dates (skipped): {len(invalid_dates)}")
        
        if future_dates:
            print(f"\n⏳ Skipping future dates (first 5): {[s for s, d in future_dates[:5]]}")
        if invalid_dates:
            print(f"\n❌ Skipping invalid dates: {[s for s, d in invalid_dates]}")
        
        return valid_dates

    def download_all_1m_launch_data(self, delay=0.3):
        """Download 4 hours of 1-minute data from launch for all coins"""
        print(f"🚀 Starting download of 4h 1-minute launch data...")
        print(f"⚠️  Each coin will have 240 candles (4 hours × 60 minutes)")
        print(f"📦 Single API call per coin (240 < 1000 limit)")
        print(f"Data will be saved to '{self.data_folder}/' folder")
        
        # Filter out future and invalid dates
        valid_dates = self.filter_valid_launch_dates()
        
        if not valid_dates:
            print("❌ No valid launch dates found!")
            return [], []
        
        print(f"\n🎯 Processing {len(valid_dates)} valid coins")
        print("=" * 80)
        
        successful_downloads = []
        failed_downloads = []
        
        try:
            for i, (symbol, launch_date_str) in enumerate(valid_dates.items(), 1):
                print(f"[{i:3d}/{len(valid_dates):3d}] ", end="")
                
                try:
                    # Check if file already exists
                    launch_date_clean = launch_date_str.replace(' ', '_').replace(':', '-')
                    filename = f"{self.data_folder}/{symbol}_1m_launch_{launch_date_clean}.csv"
                    
                    if os.path.exists(filename):
                        existing_df = pd.read_csv(filename)
                        print(f"{symbol}: ✅ Already exists ({len(existing_df)} candles)")
                        successful_downloads.append(symbol)
                        continue
                    
                    # Download 4h 1-minute data from launch
                    klines_data = self.download_4h_minutes_from_launch(symbol, launch_date_str)
                    
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
                    
                    # Rate limiting between coins
                    if delay > 0 and i < len(valid_dates):
                        time.sleep(delay)
                        
                except KeyboardInterrupt:
                    print(f"\n⚠️  Download interrupted by user at coin {i}")
                    raise
                except Exception as e:
                    error_msg = str(e)
                    if "400" in error_msg or "Bad Request" in error_msg:
                        print(f"{symbol}: ❌ API Error (likely no data available)")
                    else:
                        print(f"{symbol}: ❌ Error - {error_msg}")
                    failed_downloads.append(symbol)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Download interrupted by user after {len(successful_downloads)} successful downloads")
        
        print("\n" + "=" * 80)
        print("📊 FINAL DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {len(successful_downloads)}")
        print(f"❌ Failed: {len(failed_downloads)}")
        print(f"📊 Total 1-minute candles downloaded: {len(successful_downloads) * 240:,}")
        
        if failed_downloads:
            print(f"\n❌ Failed symbols ({len(failed_downloads)}): {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
        
        return successful_downloads, failed_downloads
    
    def analyze_1m_launch_data(self):
        """Analyze the downloaded 1-minute launch data"""
        print("\n🔍 ANALYZING 1-MINUTE LAUNCH DATA...")
        print("=" * 80)
        
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv') and '_1m_launch_' in f]
        
        if not csv_files:
            print("No 1-minute data files found to analyze")
            return
        
        analysis_data = []
        
        for filename in csv_files:
            try:
                filepath = os.path.join(self.data_folder, filename)
                df = pd.read_csv(filepath)
                
                if len(df) > 0:
                    symbol = df['symbol'].iloc[0]
                    launch_date = df['launch_date'].iloc[0]
                    
                    # Calculate first 4h performance (using 1-minute data)
                    first_price = df['open'].iloc[0]
                    last_price = df['close'].iloc[-1]
                    high_4h = df['high'].max()
                    low_4h = df['low'].min()
                    
                    performance_4h = ((last_price - first_price) / first_price) * 100
                    max_gain_4h = ((high_4h - first_price) / first_price) * 100
                    max_loss_4h = ((low_4h - first_price) / first_price) * 100
                    
                    total_volume = df['volume'].sum()
                    
                    # Calculate 15-minute, 30-minute, 60-minute, 120-minute performance
                    performance_15m = ((df['close'].iloc[14] - first_price) / first_price) * 100 if len(df) > 14 else 0
                    performance_30m = ((df['close'].iloc[29] - first_price) / first_price) * 100 if len(df) > 29 else 0
                    performance_60m = ((df['close'].iloc[59] - first_price) / first_price) * 100 if len(df) > 59 else 0
                    performance_120m = ((df['close'].iloc[119] - first_price) / first_price) * 100 if len(df) > 119 else 0
                    performance_180m = ((df['close'].iloc[179] - first_price) / first_price) * 100 if len(df) > 179 else 0
                    
                    analysis_data.append({
                        'symbol': symbol,
                        'launch_date': launch_date,
                        'first_price': first_price,
                        'last_price_4h': last_price,
                        'performance_15m': performance_15m,
                        'performance_30m': performance_30m,
                        'performance_60m': performance_60m,
                        'performance_120m': performance_120m,
                        'performance_180m': performance_180m,
                        'performance_4h': performance_4h,
                        'max_gain_4h': max_gain_4h,
                        'max_loss_4h': max_loss_4h,
                        'high_4h': high_4h,
                        'low_4h': low_4h,
                        'total_volume_4h': total_volume,
                        'candles_count': len(df)
                    })
                    
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
        
        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            
            print(f"📈 1-MINUTE LAUNCH DATA ANALYSIS:")
            print(f"   Total coins analyzed: {len(analysis_df)}")
            print(f"   Average 4h performance: {analysis_df['performance_4h'].mean():.2f}%")
            print(f"   Average 15m performance: {analysis_df['performance_15m'].mean():.2f}%")
            print(f"   Average 30m performance: {analysis_df['performance_30m'].mean():.2f}%")
            print(f"   Average 60m performance: {analysis_df['performance_60m'].mean():.2f}%")
            print(f"   Average 120m performance: {analysis_df['performance_120m'].mean():.2f}%")
            print(f"   Average 180m performance: {analysis_df['performance_180m'].mean():.2f}%")
            print(f"   Best 4h performance: {analysis_df['performance_4h'].max():.2f}% ({analysis_df.loc[analysis_df['performance_4h'].idxmax(), 'symbol']})")
            print(f"   Worst 4h performance: {analysis_df['performance_4h'].min():.2f}% ({analysis_df.loc[analysis_df['performance_4h'].idxmin(), 'symbol']})")
            
            print(f"\n🏆 TOP 10 BEST 4H PERFORMERS (1-minute data):")
            top_performers = analysis_df.nlargest(10, 'performance_4h')[['symbol', 'performance_15m', 'performance_30m', 'performance_60m', 'performance_120m', 'performance_4h', 'launch_date']]
            print(top_performers.to_string(index=False))
            
            print(f"\n📉 TOP 10 WORST 4H PERFORMERS (1-minute data):")
            worst_performers = analysis_df.nsmallest(10, 'performance_4h')[['symbol', 'performance_15m', 'performance_30m', 'performance_60m', 'performance_120m', 'performance_4h', 'launch_date']]
            print(worst_performers.to_string(index=False))
            
            # Save analysis
            analysis_filename = f"{self.data_folder}/1m_launch_analysis.csv"
            analysis_df.to_csv(analysis_filename, index=False)
            print(f"\n💾 Analysis saved to {analysis_filename}")
            
            return analysis_df
        
        return None
    
    def show_sample_data(self, num_samples=3):
        """Show sample data for a few coins"""
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv') and 'analysis' not in f and '_1m_launch_' in f]
        
        if not csv_files:
            print("No 1-minute sample data to show")
            return
        
        print(f"\n📋 SAMPLE 1-MINUTE DATA (First {num_samples} coins):")
        print("=" * 80)
        
        for filename in csv_files[:num_samples]:
            try:
                filepath = os.path.join(self.data_folder, filename)
                df = pd.read_csv(filepath)
                
                symbol = df['symbol'].iloc[0]
                launch_date = df['launch_date'].iloc[0]
                
                print(f"\n🪙 {symbol} - Launch: {launch_date}")
                print(f"First 30 minutes of trading (1-minute intervals):")
                sample_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'minutes_from_launch']].head(30)
                print(sample_df.to_string(index=False))
                
                print(f"\nSummary for {symbol}:")
                print(f"   Total candles: {len(df)}")
                print(f"   First price: ${df['open'].iloc[0]:.6f}")
                print(f"   Last price: ${df['close'].iloc[-1]:.6f}")
                print(f"   4h performance: {((df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]) * 100:.2f}%")
                
            except Exception as e:
                print(f"Error showing sample for {filename}: {e}")

def main():
    print("🚀 === 1-Minute Launch Data Downloader ===")
    print("Downloads 4 hours of 1-minute candlestick data from each coin's launch date")
    print("⚠️  Each coin will have 240 candlesticks (4 hours × 60 minutes)")
    print("=" * 80)
    
    downloader = OneMinuteDataDownloader()
    
    if not downloader.launch_dates:
        print("❌ No launch dates available. Please run coin_launch_dates.py first.")
        return
    
    print(f"📋 Loaded {len(downloader.launch_dates)} total coins from launch dates file")
    print(f"📦 Single API call per coin (240 < 1000 limit) - No batching needed!")
    print(f"💾 Data will be saved to: {os.path.abspath(downloader.data_folder)}/")
    
    proceed = input(f"\n🤔 Start download? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Download cancelled.")
        return
    
    # Download all 1-minute launch data
    successful, failed = downloader.download_all_1m_launch_data(delay=0.3)
    
    if successful:
        # Show sample data
        downloader.show_sample_data(2)
        
        # Analyze 1-minute launch performance
        analysis_df = downloader.analyze_1m_launch_data()
        
        print(f"\n✅ 1-minute launch data download complete!")
        print(f"📁 Data saved in: {os.path.abspath(downloader.data_folder)}/")
        print(f"🎯 Ready for trading analysis with {len(successful)} coins")
        print(f"📊 Total 1-minute candles downloaded: {len(successful) * 240:,}")
    else:
        print("❌ No successful downloads")

if __name__ == "__main__":
    main()
