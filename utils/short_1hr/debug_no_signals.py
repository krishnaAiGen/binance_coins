#!/usr/bin/env python3

import pandas as pd
import os
from datetime import datetime

def debug_no_signals():
    """Debug why no signals are being detected"""
    
    data_folder = "/Users/krishnayadav/Documents/test_projects/binance_coins/utils/short_1hr/data"
    
    print("🔍 === Debugging No Signals Issue ===")
    
    # Get first few files
    csv_files = [f for f in os.listdir(data_folder) 
                if f.endswith('.csv') and '_1h_7d_' in f and not f.endswith('_backup.csv')]
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Test first 5 files
    for i, filename in enumerate(csv_files[:5]):
        print(f"\n📊 File {i+1}: {filename}")
        filepath = os.path.join(data_folder, filename)
        
        try:
            df = pd.read_csv(filepath)
            symbol = filename.split('_1h_7d_')[0]
            
            print(f"   Symbol: {symbol}")
            print(f"   Rows: {len(df)}")
            
            # Check if we have enough data for start_hour=2
            start_hour = 2
            signal_index = start_hour - 2  # Check hour 0 for signal (0-indexed)
            entry_index = start_hour - 1   # Enter at hour 1 (0-indexed)
            
            print(f"   Signal index: {signal_index} (hour {signal_index})")
            print(f"   Entry index: {entry_index} (hour {entry_index})")
            
            if signal_index < 0 or entry_index >= len(df):
                print(f"   ❌ Not enough data: need at least {start_hour} hours")
                continue
            
            # Check signal candle data
            signal_candle = df.iloc[signal_index]
            print(f"   Signal candle (hour {signal_index}):")
            print(f"     Close: {signal_candle['close']}")
            print(f"     EMA5: {signal_candle.get('ema_5', 'MISSING!')}")
            
            if 'ema_5' not in df.columns:
                print(f"   ❌ Missing EMA5 column!")
                continue
            
            ema_value = signal_candle['ema_5']
            close_value = signal_candle['close']
            
            if pd.isna(ema_value):
                print(f"   ❌ EMA5 is NaN")
                continue
            
            # Check signal condition
            signal = close_value < ema_value
            print(f"   Signal check: {close_value:.6f} < {ema_value:.6f} = {signal}")
            
            if signal:
                print(f"   ✅ SHORT SIGNAL DETECTED!")
                # Check entry candle
                entry_candle = df.iloc[entry_index]
                print(f"   Entry candle (hour {entry_index}):")
                print(f"     Open: {entry_candle['open']} (entry price)")
                print(f"     Timestamp: {entry_candle['timestamp']}")
            else:
                print(f"   🚫 No signal: close >= EMA5")
            
            # Show first few hours data
            print(f"   First 4 hours:")
            for h in range(min(4, len(df))):
                row = df.iloc[h]
                close_val = row['close']
                ema5_val = row.get('ema_5', 'N/A')
                signal_val = "SHORT" if not pd.isna(ema5_val) and close_val < ema5_val else "NO"
                print(f"     Hour {h}: Close={close_val:.6f}, EMA5={ema5_val}, Signal={signal_val}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Debug complete!")

if __name__ == "__main__":
    debug_no_signals()
