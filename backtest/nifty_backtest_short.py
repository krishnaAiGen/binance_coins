#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================
TIMEOUT_MINUTES = 5          # Maximum time to hold position
TARGET_POINTS = 2            # Points to drop for profit (SHORT position)
STOP_LOSS_POINTS = 2         # Points to rise for stop loss (SHORT position)
EMA_FAST_PERIOD = 9          # Fast EMA period
EMA_SLOW_PERIOD = 15         # Slow EMA period

def load_and_prepare_data(file_path):
    """Load CSV data and prepare it for backtesting"""
    print(f"Loading data from: {file_path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding}: {e}")
            continue
    
    if df is None:
        raise Exception("Could not load the CSV file with any encoding")
    
    print(f"Raw data shape: {df.shape}")
    print(f"Raw columns: {list(df.columns)}")
    print("First few rows:")
    print(df.head())
    
    # Find date column (flexible naming)
    date_columns = ['date', 'Date', 'timestamp', 'Timestamp', 'datetime', 'DateTime']
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise Exception(f"No date column found. Available columns: {list(df.columns)}")
    
    # Ensure we have OHLC columns (case insensitive)
    required_cols = ['open', 'high', 'low', 'close']
    col_mapping = {}
    
    for req_col in required_cols:
        found = False
        for col in df.columns:
            if col.lower() == req_col:
                col_mapping[col] = req_col
                found = True
                break
        if not found:
            raise Exception(f"Required column '{req_col}' not found")
    
    # Rename columns to lowercase
    df = df.rename(columns=col_mapping)
    
    # Convert date column
    df['timestamp'] = pd.to_datetime(df[date_col])
    
    # Ensure numeric data types
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN values
    df = df.dropna(subset=numeric_cols + ['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("Data loaded successfully!")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total rows: {len(df)}")
    
    return df

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def find_ema_crossovers(df, fast_period, slow_period):
    """Find EMA crossover points where fast EMA crosses above slow EMA"""
    # Calculate EMAs
    df[f'ema_{fast_period}'] = calculate_ema(df['close'], fast_period)
    df[f'ema_{slow_period}'] = calculate_ema(df['close'], slow_period)
    
    # Find crossovers (fast crosses above slow)
    df['ema_diff'] = df[f'ema_{fast_period}'] - df[f'ema_{slow_period}']
    df['ema_diff_prev'] = df['ema_diff'].shift(1)
    
    # Crossover occurs when previous diff was negative and current is positive
    df['crossover'] = (df['ema_diff_prev'] < 0) & (df['ema_diff'] > 0)
    
    return df

def simulate_short_trade(df, entry_idx, target_points, stop_loss_points, timeout_minutes):
    """
    Simulate a SHORT trade from entry point
    - Entry: Sell at entry_price
    - Target: Buy back when price drops by target_points (profit)
    - Stop Loss: Buy back when price rises by stop_loss_points (loss)
    """
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row['open']  # Use next candle's open price
    entry_time = entry_row['timestamp']
    
    # Calculate target and stop loss prices
    target_price = entry_price - target_points  # Profit when price drops
    stop_loss_price = entry_price + stop_loss_points  # Loss when price rises
    
    # Calculate timeout time
    timeout_time = entry_time + timedelta(minutes=timeout_minutes)
    
    # Look for exit conditions in subsequent candles
    for i in range(entry_idx, len(df)):
        current_row = df.iloc[i]
        current_time = current_row['timestamp']
        
        # Check timeout first
        if current_time >= timeout_time:
            return {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': current_row['close'],
                'profit': entry_price - current_row['close'],  # SHORT: profit when exit < entry
                'duration_minutes': (current_time - entry_time).total_seconds() / 60,
                'exit_reason': 'Timeout',
                'ema_fast_at_entry': entry_row[f'ema_{EMA_FAST_PERIOD}'],
                'ema_slow_at_entry': entry_row[f'ema_{EMA_SLOW_PERIOD}']
            }
        
        # Check stop loss (price went up - bad for short)
        if current_row['high'] >= stop_loss_price:
            return {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': stop_loss_price,
                'profit': entry_price - stop_loss_price,  # Will be negative (loss)
                'duration_minutes': (current_time - entry_time).total_seconds() / 60,
                'exit_reason': 'Stop Loss',
                'ema_fast_at_entry': entry_row[f'ema_{EMA_FAST_PERIOD}'],
                'ema_slow_at_entry': entry_row[f'ema_{EMA_SLOW_PERIOD}']
            }
        
        # Check target (price went down - good for short)
        if current_row['low'] <= target_price:
            return {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': target_price,
                'profit': entry_price - target_price,  # Will be positive (profit)
                'duration_minutes': (current_time - entry_time).total_seconds() / 60,
                'exit_reason': 'Target Hit',
                'ema_fast_at_entry': entry_row[f'ema_{EMA_FAST_PERIOD}'],
                'ema_slow_at_entry': entry_row[f'ema_{EMA_SLOW_PERIOD}']
            }
    
    # If no exit condition met, exit at last available price
    last_row = df.iloc[-1]
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': last_row['timestamp'],
        'exit_price': last_row['close'],
        'profit': entry_price - last_row['close'],
        'duration_minutes': (last_row['timestamp'] - entry_time).total_seconds() / 60,
        'exit_reason': 'End of Data',
        'ema_fast_at_entry': entry_row[f'ema_{EMA_FAST_PERIOD}'],
        'ema_slow_at_entry': entry_row[f'ema_{EMA_SLOW_PERIOD}']
    }

def backtest_short_strategy(df, start_date, end_date, fast_period, slow_period, target_points, stop_loss_points, timeout_minutes):
    """Run backtest for SHORT strategy on EMA crossovers"""
    
    # Filter data for backtest period
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    test_df = df[mask].copy().reset_index(drop=True)
    
    if len(test_df) == 0:
        print("No data available for the specified date range")
        return pd.DataFrame(), {}
    
    print(f"Backtesting SHORT strategy from {start_date} to {end_date}")
    print(f"Using EMA({fast_period}, {slow_period}), Target: {target_points} points DOWN, Stop Loss: {stop_loss_points} points UP, Timeout: {timeout_minutes} min")
    print("=" * 60)
    
    # Find crossovers
    test_df = find_ema_crossovers(test_df, fast_period, slow_period)
    crossover_indices = test_df[test_df['crossover']].index.tolist()
    
    print(f"Found {len(crossover_indices)} EMA crossover signals")
    
    if len(crossover_indices) == 0:
        print("No crossover signals found in the specified period")
        return pd.DataFrame(), {}
    
    trades = []
    
    for i, cross_idx in enumerate(crossover_indices):
        # Enter SHORT trade at next candle after crossover
        if cross_idx + 1 < len(test_df):
            entry_idx = cross_idx + 1
            
            trade_result = simulate_short_trade(
                test_df, entry_idx, target_points, stop_loss_points, timeout_minutes
            )
            
            # Add additional info
            trade_result['ema_fast_period'] = fast_period
            trade_result['ema_slow_period'] = slow_period
            
            trades.append(trade_result)
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return trades_df, {}
    
    # Calculate statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit'] > 0])
    losing_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = trades_df['profit'].sum()
    avg_profit = trades_df['profit'].mean()
    
    # Count exit reasons
    target_hits = len(trades_df[trades_df['exit_reason'] == 'Target Hit'])
    stop_losses = len(trades_df[trades_df['exit_reason'] == 'Stop Loss'])
    timeouts = len(trades_df[trades_df['exit_reason'] == 'Timeout'])
    
    avg_duration = trades_df['duration_minutes'].mean()
    
    stats = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'target_hits': target_hits,
        'stop_losses': stop_losses,
        'timeouts': timeouts,
        'avg_duration': avg_duration
    }
    
    print(f"Tested EMA({fast_period}, {slow_period}): {total_trades} trades, Total Profit: {total_profit:.2f}, Win Rate: {win_rate:.1f}%, Target Hits: {target_hits}, Stop Loss: {stop_losses}, Timeouts: {timeouts}")
    
    return trades_df, stats

def print_detailed_results(trades_df, stats):
    """Print detailed backtest results"""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print("📊 SHORT TRADE STATISTICS:")
    print(f"   Total trades executed: {stats['total_trades']}")
    print(f"   Winning trades: {stats['winning_trades']}")
    print(f"   Losing trades: {stats['losing_trades']}")
    print(f"   Win rate: {stats['win_rate']:.1f}%")
    
    print(f"\n💰 PROFIT & LOSS:")
    profit_trades = trades_df[trades_df['profit'] > 0]['profit'].sum()
    loss_trades = trades_df[trades_df['profit'] <= 0]['profit'].sum()
    print(f"   Total PROFIT: {profit_trades:.2f} points")
    print(f"   Total LOSS: {loss_trades:.2f} points")
    print(f"   Net Profit: {stats['total_profit']:.2f} points")
    print(f"   Average profit per trade: {stats['avg_profit']:.2f} points")
    
    print(f"\n📈 EXIT ANALYSIS:")
    print(f"   Target hits: {stats['target_hits']}")
    print(f"   Stop losses: {stats['stop_losses']}")
    print(f"   {TIMEOUT_MINUTES}-min timeouts: {stats['timeouts']}")
    print(f"   Average trade duration: {stats['avg_duration']:.1f} minutes")
    
    print(f"\n📋 FIRST 10 SHORT TRADES:")
    display_cols = ['entry_time', 'entry_price', 'exit_price', 'profit', 'duration_minutes', 'exit_reason']
    print(trades_df[display_cols].head(10).to_string())

def main():
    print("=== NIFTY 50 EMA Crossover SHORT Strategy Backtesting ===")
    print("Configuration:")
    print(f"  - Timeout: {TIMEOUT_MINUTES} minutes")
    print(f"  - Target: {TARGET_POINTS} points DOWN (profit)")
    print(f"  - Stop Loss: {STOP_LOSS_POINTS} points UP (loss)")
    print(f"  - EMA Fast: {EMA_FAST_PERIOD}, EMA Slow: {EMA_SLOW_PERIOD}")
    print("=" * 60)
    
    # Load data
    file_path = "/Users/krishnayadav/Downloads/nifty_july_data/NIFTY 50_minute_data.csv"
    df = load_and_prepare_data(file_path)
    
    # Define backtest period
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    
    # Run backtest
    trades_df, stats = backtest_short_strategy(
        df, start_date, end_date, 
        EMA_FAST_PERIOD, EMA_SLOW_PERIOD, 
        TARGET_POINTS, STOP_LOSS_POINTS, TIMEOUT_MINUTES
    )
    
    if len(trades_df) > 0:
        print(f"\nBest parameters: EMA({EMA_FAST_PERIOD}, {EMA_SLOW_PERIOD})")
        print(f"Total trades: {stats['total_trades']}")
        print(f"Total profit: {stats['total_profit']:.2f}")
        print(f"Win rate: {stats['win_rate']:.1f}%")
        print(f"Target hits: {stats['target_hits']}, Timeouts: {stats['timeouts']}")
        print(f"Average duration: {stats['avg_duration']:.1f} minutes")
        
        # Show detailed results with all trades
        print(f"\nDetailed SHORT trade results:")
        trades_df['bb_distance_at_entry'] = np.nan  # Placeholder
        print(trades_df[['entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 
                        'duration_minutes', 'exit_reason', 'ema_fast_at_entry', 'ema_slow_at_entry',
                        'bb_distance_at_entry', 'ema_fast_period', 'ema_slow_period']].to_string())
        
        # Print summary
        print_detailed_results(trades_df, stats)
    else:
        print("No trades executed in the specified period")

if __name__ == "__main__":
    main() 