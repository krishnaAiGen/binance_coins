#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EMACalculator:
    def __init__(self, data_folder="/Users/krishnayadav/Documents/test_projects/binance_coins/utils/short_1hr/data"):
        self.data_folder = data_folder
        self.ema_periods = [5, 10, 20, 50, 100]  # 5 different EMA periods
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average for given period"""
        return data.ewm(span=period, adjust=False).mean()
    
    def add_ema_indicators(self, df):
        """Add 5 EMA indicators to the dataframe"""
        if 'close' not in df.columns:
            print("❌ Error: 'close' column not found in dataframe")
            return df
        
        # Calculate EMAs based on closing price
        for period in self.ema_periods:
            ema_column = f'ema_{period}'
            df[ema_column] = self.calculate_ema(df['close'], period)
        
        # Add EMA trend signals
        df['ema_5_above_10'] = (df['ema_5'] > df['ema_10']).astype(int)
        df['ema_5_above_20'] = (df['ema_5'] > df['ema_20']).astype(int)
        df['ema_10_above_20'] = (df['ema_10'] > df['ema_20']).astype(int)
        df['ema_20_above_50'] = (df['ema_20'] > df['ema_50']).astype(int)
        df['ema_50_above_100'] = (df['ema_50'] > df['ema_100']).astype(int)
        
        # Add bullish/bearish signals
        # Bullish: Short EMAs above Long EMAs
        df['ema_bullish_signal'] = (
            (df['ema_5'] > df['ema_10']) & 
            (df['ema_10'] > df['ema_20']) & 
            (df['ema_20'] > df['ema_50'])
        ).astype(int)
        
        # Bearish: Short EMAs below Long EMAs  
        df['ema_bearish_signal'] = (
            (df['ema_5'] < df['ema_10']) & 
            (df['ema_10'] < df['ema_20']) & 
            (df['ema_20'] < df['ema_50'])
        ).astype(int)
        
        # EMA crossover signals
        df['ema_5_10_crossover'] = ((df['ema_5'] > df['ema_10']) & (df['ema_5'].shift(1) <= df['ema_10'].shift(1))).astype(int)
        df['ema_10_20_crossover'] = ((df['ema_10'] > df['ema_20']) & (df['ema_10'].shift(1) <= df['ema_20'].shift(1))).astype(int)
        
        return df
    
    def process_file(self, filepath):
        """Process a single CSV file to add EMA indicators"""
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            if len(df) == 0:
                print(f"❌ {os.path.basename(filepath)}: Empty file")
                return False
            
            # Check if EMAs already exist
            if 'ema_5' in df.columns:
                print(f"⚠️  {os.path.basename(filepath)}: EMAs already exist, skipping")
                return True
            
            # Add EMA indicators
            df_with_ema = self.add_ema_indicators(df.copy())
            
            # Create backup
            backup_filepath = filepath.replace('.csv', '_backup.csv')
            if not os.path.exists(backup_filepath):
                df.to_csv(backup_filepath, index=False)
            
            # Save updated file
            df_with_ema.to_csv(filepath, index=False)
            
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown'
            print(f"✅ {symbol}: Added {len(self.ema_periods)} EMAs + signals ({len(df_with_ema)} rows)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(filepath)}: {e}")
            return False
    
    def process_all_files(self):
        """Process all CSV files in the data folder"""
        print("🚀 === EMA Indicator Calculator ===")
        print(f"Adding {len(self.ema_periods)} EMA indicators: {self.ema_periods}")
        print(f"Processing files in: {self.data_folder}")
        print("=" * 80)
        
        if not os.path.exists(self.data_folder):
            print(f"❌ Error: Data folder not found: {self.data_folder}")
            return
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.data_folder) 
                    if f.endswith('.csv') and not f.endswith('_backup.csv') and 'analysis' not in f.lower()]
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.data_folder}")
            return
        
        print(f"📁 Found {len(csv_files)} CSV files to process")
        
        # Confirm before processing
        proceed = input(f"\n🤔 Add EMAs to all {len(csv_files)} files? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Operation cancelled.")
            return
        
        successful = 0
        failed = 0
        skipped = 0
        
        print("\n📊 Processing files...")
        print("=" * 80)
        
        for i, filename in enumerate(csv_files, 1):
            filepath = os.path.join(self.data_folder, filename)
            print(f"[{i:3d}/{len(csv_files):3d}] ", end="")
            
            result = self.process_file(filepath)
            
            if result is True:
                successful += 1
            elif result is False:
                failed += 1
            else:
                skipped += 1
        
        print("\n" + "=" * 80)
        print("📊 FINAL PROCESSING SUMMARY")
        print("=" * 80)
        print(f"✅ Successfully processed: {successful}")
        print(f"⚠️  Skipped (already had EMAs): {skipped}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Total files with EMAs: {successful + skipped}")
        
        if successful > 0:
            print(f"\n🎯 EMA INDICATORS ADDED:")
            print(f"   • Basic EMAs: {', '.join([f'EMA-{p}' for p in self.ema_periods])}")
            print(f"   • Trend Signals: ema_5_above_10, ema_5_above_20, etc.")
            print(f"   • Bull/Bear Signals: ema_bullish_signal, ema_bearish_signal")
            print(f"   • Crossover Signals: ema_5_10_crossover, ema_10_20_crossover")
            print(f"\n💾 Backup files created with '_backup.csv' suffix")
            print(f"📁 Files processed in: {self.data_folder}")
    
    def show_sample_ema_data(self, num_samples=2):
        """Show sample EMA data for verification"""
        csv_files = [f for f in os.listdir(self.data_folder) 
                    if f.endswith('.csv') and not f.endswith('_backup.csv') and 'analysis' not in f.lower()]
        
        if not csv_files:
            print("No files to show sample data from")
            return
        
        print(f"\n📋 SAMPLE EMA DATA (First {num_samples} files):")
        print("=" * 80)
        
        for filename in csv_files[:num_samples]:
            try:
                filepath = os.path.join(self.data_folder, filename)
                df = pd.read_csv(filepath)
                
                if 'ema_5' not in df.columns:
                    print(f"⚠️  {filename}: No EMA data found")
                    continue
                
                symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown'
                
                print(f"\n🪙 {symbol} - EMA Sample (Last 10 rows):")
                
                # Show relevant columns
                ema_cols = ['timestamp', 'close'] + [f'ema_{p}' for p in self.ema_periods]
                if 'hours_from_launch' in df.columns:
                    ema_cols.insert(1, 'hours_from_launch')
                
                sample_df = df[ema_cols].tail(10)
                print(sample_df.to_string(index=False))
                
                # Show signals for last row
                last_row = df.iloc[-1]
                print(f"\nSignals for {symbol} (Latest):")
                print(f"   🔥 Bullish Signal: {bool(last_row.get('ema_bullish_signal', 0))}")
                print(f"   🔻 Bearish Signal: {bool(last_row.get('ema_bearish_signal', 0))}")
                print(f"   📈 EMA 5 > 10: {bool(last_row.get('ema_5_above_10', 0))}")
                print(f"   📈 EMA 10 > 20: {bool(last_row.get('ema_10_above_20', 0))}")
                
            except Exception as e:
                print(f"Error showing sample for {filename}: {e}")

def main():
    print("🚀 === EMA Indicator Calculator ===")
    print("Adds 5 EMA indicators (5, 10, 20, 50, 100) plus trading signals")
    print("=" * 80)
    
    calculator = EMACalculator()
    
    # Process all files
    calculator.process_all_files()
    
    # Show sample data
    calculator.show_sample_ema_data(2)

if __name__ == "__main__":
    main()