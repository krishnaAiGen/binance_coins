import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class NFT50EMABacktester:
    def __init__(self, data, date_column='timestamp'):
        """
        Initialize the backtester with NFT50 data
        
        Parameters:
        data (pd.DataFrame): DataFrame with OHLC data
        date_column (str): Name of the date/timestamp column (default: 'timestamp')
        """
        self.data = data.copy()
        
        # Handle different date column names
        if date_column in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data[date_column])
        elif 'date' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['date'])
        elif 'Date' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['Date'])
        elif 'TIME' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['TIME'])
        else:
            # If no recognized date column, assume first column is date
            first_col = self.data.columns[0]
            print(f"Warning: Using '{first_col}' as date column")
            self.data['timestamp'] = pd.to_datetime(self.data[first_col])
        
        # Ensure required OHLC columns exist (handle case variations)
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in self.data.columns:
                # Try capitalized version
                if col.capitalize() in self.data.columns:
                    self.data[col] = self.data[col.capitalize()]
                elif col.upper() in self.data.columns:
                    self.data[col] = self.data[col.upper()]
                else:
                    raise ValueError(f"Required column '{col}' not found in data")
        
        # Add volume column if it doesn't exist
        if 'volume' not in self.data.columns:
            self.data['volume'] = 1000  # Default volume
        
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        self.results = []
        
        print(f"Data loaded successfully!")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"Total rows: {len(self.data)}")
        
    def calculate_indicators(self, ema_fast=9, ema_slow=15, bb_period=20, bb_std=2):
        """
        Calculate technical indicators
        """
        df = self.data.copy()
        
        # Calculate EMAs
        df[f'EMA_{ema_fast}'] = df['close'].ewm(span=ema_fast).mean()
        df[f'EMA_{ema_slow}'] = df['close'].ewm(span=ema_slow).mean()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Distance'] = df['BB_Upper'] - df['BB_Middle']
        
        # Identify EMA crossovers
        df['EMA_Cross'] = 0
        df.loc[(df[f'EMA_{ema_fast}'] > df[f'EMA_{ema_slow}']) & 
               (df[f'EMA_{ema_fast}'].shift(1) <= df[f'EMA_{ema_slow}'].shift(1)), 'EMA_Cross'] = 1
        
        return df
    
    def backtest_strategy(self, start_date, end_date, ema_fast=9, ema_slow=15, 
                         target_points=2, bb_period=20, bb_std=2, timeout_minutes=10, stop_loss_points=2):
        """
        Backtest the EMA crossover strategy
        
        Parameters:
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        ema_fast (int): Fast EMA period
        ema_slow (int): Slow EMA period
        target_points (float): Profit target in points
        timeout_minutes (int): Maximum time to hold position before force exit
        stop_loss_points (float): Stop loss in points below entry price
        """
        
        # Filter data by date range
        mask = (self.data['timestamp'] >= start_date) & (self.data['timestamp'] <= end_date)
        df = self.data[mask].copy().reset_index(drop=True)
        
        if len(df) < max(ema_fast, ema_slow, bb_period) + 10:
            return pd.DataFrame(), {}
        
        # Calculate indicators
        df = self.calculate_indicators_on_df(df, ema_fast, ema_slow, bb_period, bb_std)
        
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        entry_idx = 0
        entry_ema_fast = 0
        entry_ema_slow = 0
        entry_bb_distance = 0
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # Check for entry signal (EMA crossover at index i-1, enter at index i)
            if not in_position and i > 0:
                prev_row = df.iloc[i-1]
                if prev_row['EMA_Cross'] == 1:
                    # Enter position at current candle's open (i+1 from crossover)
                    in_position = True
                    entry_price = current_row['open']
                    entry_time = current_row['timestamp']
                    entry_idx = i
                    entry_ema_fast = current_row[f'EMA_{ema_fast}']
                    entry_ema_slow = current_row[f'EMA_{ema_slow}']
                    entry_bb_distance = current_row['BB_Distance']
                    # Uncomment below for detailed debugging
                    # print(f"Entry at index {i}: Price {entry_price:.2f}, Time {entry_time}")
            
            # Check for exit conditions
            if in_position:
                duration_minutes = (current_row['timestamp'] - entry_time).total_seconds() / 60
                
                # Check if stop loss hit (priority 1)
                if current_row['low'] <= entry_price - stop_loss_points:
                    # Exit at stop loss price
                    exit_price = entry_price - stop_loss_points
                    exit_time = current_row['timestamp']
                    profit = -stop_loss_points
                    exit_reason = "Stop Loss"
                    
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'profit': profit,
                        'duration_minutes': duration_minutes,
                        'exit_reason': exit_reason,
                        'ema_fast_at_entry': entry_ema_fast,
                        'ema_slow_at_entry': entry_ema_slow,
                        'bb_distance_at_entry': entry_bb_distance,
                        'ema_fast_period': ema_fast,
                        'ema_slow_period': ema_slow
                    }
                    trades.append(trade)
                    in_position = False
                    # Uncomment below for detailed debugging
                    # print(f"Exit at index {i}: Price {exit_price:.2f}, Profit {profit:.2f}, Duration {duration_minutes:.1f}min, Reason: {exit_reason}")
                
                # Check if target reached (priority 2)
                elif current_row['high'] >= entry_price + target_points:
                    # Exit at target price
                    exit_price = entry_price + target_points
                    exit_time = current_row['timestamp']
                    profit = target_points
                    exit_reason = "Target Hit"
                    
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'profit': profit,
                        'duration_minutes': duration_minutes,
                        'exit_reason': exit_reason,
                        'ema_fast_at_entry': entry_ema_fast,
                        'ema_slow_at_entry': entry_ema_slow,
                        'bb_distance_at_entry': entry_bb_distance,
                        'ema_fast_period': ema_fast,
                        'ema_slow_period': ema_slow
                    }
                    trades.append(trade)
                    in_position = False
                    # Uncomment below for detailed debugging
                    # print(f"Exit at index {i}: Price {exit_price:.2f}, Profit {profit:.2f}, Duration {duration_minutes:.1f}min, Reason: {exit_reason}")
                
                # Check if timeout reached (10 minutes)
                elif duration_minutes >= timeout_minutes:
                    # Exit at current close price after timeout
                    exit_price = current_row['close']  # Exit at close price
                    exit_time = current_row['timestamp']
                    profit = exit_price - entry_price
                    exit_reason = f"{timeout_minutes}min Timeout"
                    
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'profit': profit,
                        'duration_minutes': duration_minutes,
                        'exit_reason': exit_reason,
                        'ema_fast_at_entry': entry_ema_fast,
                        'ema_slow_at_entry': entry_ema_slow,
                        'bb_distance_at_entry': entry_bb_distance,
                        'ema_fast_period': ema_fast,
                        'ema_slow_period': ema_slow
                    }
                    trades.append(trade)
                    in_position = False
                    # Uncomment below for detailed debugging
                    # print(f"Exit at index {i}: Price {exit_price:.2f}, Profit {profit:.2f}, Duration {duration_minutes:.1f}min, Reason: {exit_reason}")
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate summary statistics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] <= 0]
            target_hit_trades = trades_df[trades_df['exit_reason'] == 'Target Hit']
            stop_loss_trades = trades_df[trades_df['exit_reason'] == 'Stop Loss']
            timeout_trades = trades_df[trades_df['exit_reason'].str.contains('Timeout')]
            
            total_profit = trades_df['profit'].sum()
            total_loss = losing_trades['profit'].sum() if len(losing_trades) > 0 else 0
            total_gain = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
            
            summary = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'target_hit_trades': len(target_hit_trades),
                'stop_loss_trades': len(stop_loss_trades),
                'timeout_trades': len(timeout_trades),
                'total_profit': total_profit,
                'total_gain': total_gain,
                'total_loss': abs(total_loss),
                'avg_profit_per_trade': trades_df['profit'].mean(),
                'avg_duration_minutes': trades_df['duration_minutes'].mean(),
                'win_rate': (len(winning_trades) / len(trades_df)) * 100,
                'profit_factor': total_gain / abs(total_loss) if total_loss != 0 else float('inf') if total_gain > 0 else 0,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'start_date': start_date,
                'end_date': end_date,
                'timeout_minutes': timeout_minutes
            }
        else:
            summary = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'target_hit_trades': 0,
                'stop_loss_trades': 0,
                'timeout_trades': 0,
                'total_profit': 0,
                'total_gain': 0,
                'total_loss': 0,
                'avg_profit_per_trade': 0,
                'avg_duration_minutes': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'start_date': start_date,
                'end_date': end_date,
                'timeout_minutes': timeout_minutes
            }
        
        return trades_df, summary
    
    def calculate_indicators_on_df(self, df, ema_fast, ema_slow, bb_period, bb_std):
        """Helper function to calculate indicators on a dataframe"""
        # Calculate EMAs
        df[f'EMA_{ema_fast}'] = df['close'].ewm(span=ema_fast).mean()
        df[f'EMA_{ema_slow}'] = df['close'].ewm(span=ema_slow).mean()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Distance'] = df['BB_Upper'] - df['BB_Middle']
        
        # Identify EMA crossovers
        df['EMA_Cross'] = 0
        df.loc[(df[f'EMA_{ema_fast}'] > df[f'EMA_{ema_slow}']) & 
               (df[f'EMA_{ema_fast}'].shift(1) <= df[f'EMA_{ema_slow}'].shift(1)), 'EMA_Cross'] = 1
        
        return df
    
    def grid_search(self, start_date, end_date, ema_fast_range, ema_slow_range, target_points=2, timeout_minutes=10, stop_loss_points=2):
        """
        Perform grid search optimization
        
        Parameters:
        start_date (str): Start date for backtesting
        end_date (str): End date for backtesting
        ema_fast_range (list): List of fast EMA periods to test
        ema_slow_range (list): List of slow EMA periods to test
        target_points (float): Profit target in points
        timeout_minutes (int): Maximum time to hold position before force exit
        stop_loss_points (float): Stop loss in points below entry price
        """
        
        results = []
        
        for ema_fast, ema_slow in product(ema_fast_range, ema_slow_range):
            if ema_fast >= ema_slow:  # Skip invalid combinations
                continue
                
            trades_df, summary = self.backtest_strategy(
                start_date, end_date, ema_fast, ema_slow, target_points, timeout_minutes=timeout_minutes, stop_loss_points=stop_loss_points
            )
            
            results.append(summary)
            print(f"Tested EMA({ema_fast}, {ema_slow}): {summary['total_trades']} trades, "
                  f"Total Profit: {summary['total_profit']:.2f}, Win Rate: {summary['win_rate']:.1f}%, "
                  f"Target Hits: {summary['target_hit_trades']}, Stop Loss: {summary['stop_loss_trades']}, Timeouts: {summary['timeout_trades']}")
        
        results_df = pd.DataFrame(results)
        
        # Sort by total profit descending
        results_df = results_df.sort_values('total_profit', ascending=False)
        
        return results_df
    
    def plot_results(self, trades_df, title="Trading Results"):
        """Plot trading results"""
        if len(trades_df) == 0:
            print("No trades to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Profit per trade
        axes[0, 0].plot(trades_df.index, trades_df['profit'].cumsum())
        axes[0, 0].set_title('Cumulative Profit')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Cumulative Profit')
        axes[0, 0].grid(True)
        
        # Trade duration distribution
        axes[0, 1].hist(trades_df['duration_minutes'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Trade Duration Distribution')
        axes[0, 1].set_xlabel('Duration (Minutes)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Exit reason pie chart
        exit_counts = trades_df['exit_reason'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        axes[0, 2].pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0, 2].set_title('Exit Reasons')
        
        # Profit distribution by exit reason
        target_profits = trades_df[trades_df['exit_reason'] == 'Target Hit']['profit']
        timeout_profits = trades_df[trades_df['exit_reason'].str.contains('Timeout')]['profit']
        
        axes[1, 0].hist([target_profits, timeout_profits], bins=20, alpha=0.7, 
                       label=['Target Hit', 'Timeout'], color=['green', 'red'])
        axes[1, 0].set_title('Profit Distribution by Exit Reason')
        axes[1, 0].set_xlabel('Profit')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Bollinger Band distance at entry
        axes[1, 1].hist(trades_df['bb_distance_at_entry'], bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_title('BB Distance at Entry Distribution')
        axes[1, 1].set_xlabel('BB Distance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        # Trades over time
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        daily_trades = trades_df.groupby('entry_date')['profit'].sum()
        axes[1, 2].plot(daily_trades.index, daily_trades.values, marker='o', markersize=4)
        axes[1, 2].set_title('Daily Profit')
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Daily Profit')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

# Example usage and testing
def run_backtest_example():
    """
    Example function showing how to use the backtester
    """
    
    # Generate sample NFT50-like data for demonstration
    # Replace this with your actual NFT50 data loading
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='1min')
    
    # Generate realistic price data
    initial_price = 100
    prices = [initial_price]
    
    for i in range(len(dates) - 1):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.05)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure price doesn't go negative
    
    # Create OHLC data
    sample_data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        open_price = close + np.random.normal(0, 0.02)
        high = max(open_price, close) + abs(np.random.normal(0, 0.01))
        low = min(open_price, close) - abs(np.random.normal(0, 0.01))
        volume = np.random.randint(1000, 10000)
        
        sample_data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(sample_data)
    
    # Initialize backtester
    backtester = NFT50EMABacktester(df)
    
    print("=== NFT50 EMA Crossover Backtesting System ===\n")
    
    # Configuration parameters
    timeout_minutes = 10
    target_points = 2
    stop_loss_points = 2
    
    # Single backtest
    print("1. Running single backtest...")
    trades_df, summary = backtester.backtest_strategy(
        start_date='2023-06-01',
        end_date='2023-12-31',
        ema_fast=9,
        ema_slow=15,
        target_points=target_points,
        timeout_minutes=timeout_minutes,
        stop_loss_points=stop_loss_points
    )
    
    print(f"Results: {summary['total_trades']} trades, Total Profit: {summary['total_profit']:.2f}")
    print(f"Win Rate: {summary['win_rate']:.1f}%, Target Hits: {summary['target_hit_trades']}, Timeouts: {summary['timeout_trades']}")
    print(f"Average Duration: {summary['avg_duration_minutes']:.1f} minutes")
    
    if len(trades_df) > 0:
        print("\nFirst 5 trades:")
        print(trades_df.head().to_string())
    
    # Grid search
    print("\n2. Running grid search optimization...")
    # ema_fast_range = [5, 9, 12, 15]
    # ema_slow_range = [15, 20, 25, 30]
    ema_fast_range = [9]
    ema_slow_range = [15]
    
    grid_results = backtester.grid_search(
        start_date='2023-06-01',
        end_date='2023-12-31',
        ema_fast_range=ema_fast_range,
        ema_slow_range=ema_slow_range,
        target_points=target_points,
        timeout_minutes=timeout_minutes,
        stop_loss_points=stop_loss_points
    )
    
    print("\nTop 5 parameter combinations:")
    print(grid_results.head().to_string())
    
    return backtester, trades_df, grid_results

# For actual usage with your NFT50 data:
def load_and_backtest_nft50(file_path, start_date, end_date, timeout_minutes=10, target_points=2, stop_loss_points=2, ema_fast=9, ema_slow=15):
    """
    Load actual NFT50 data and run backtesting
    
    Parameters:
    file_path (str): Path to your NFT50 data file (CSV)
    start_date (str): Start date for backtesting (YYYY-MM-DD)
    end_date (str): End date for backtesting (YYYY-MM-DD)
    timeout_minutes (int): Maximum time to hold position before force exit
    target_points (float): Profit target in points
    stop_loss_points (float): Stop loss in points below entry price
    ema_fast (int): Fast EMA period
    ema_slow (int): Slow EMA period
    """
    
    # Load your NFT50 data
    print(f"Loading data from: {file_path}")
    
    try:
        # Try different encoding options
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception:
            df = pd.read_csv(file_path, encoding='utf-8')
            
        print(f"Raw data shape: {df.shape}")
        print(f"Raw columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Initialize backtester (it will auto-detect the date column)
        backtester = NFT50EMABacktester(df)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your file path and CSV format")
        return None, None, None
    
    # Define parameter ranges for grid search (using provided parameters)
    ema_fast_range = [ema_fast]
    ema_slow_range = [ema_slow]
    
    print(f"Backtesting NFT50 strategy from {start_date} to {end_date}")
    print(f"Using EMA({ema_fast}, {ema_slow}), Target: {target_points} points, Stop Loss: {stop_loss_points} points, Timeout: {timeout_minutes} min")
    print("="*60)
    
    # Run grid search
    grid_results = backtester.grid_search(
        start_date=start_date,
        end_date=end_date,
        ema_fast_range=ema_fast_range,
        ema_slow_range=ema_slow_range,
        target_points=target_points,
        timeout_minutes=timeout_minutes,
        stop_loss_points=stop_loss_points
    )
    
    if len(grid_results) == 0:
        print("No valid results from grid search. Check your date range and data.")
        return backtester, pd.DataFrame(), pd.DataFrame()
    
    # Get best parameters
    best_params = grid_results.iloc[0]
    print(f"\nBest parameters: EMA({best_params['ema_fast']}, {best_params['ema_slow']})")
    print(f"Total trades: {best_params['total_trades']}")
    print(f"Total profit: {best_params['total_profit']:.2f}")
    print(f"Win rate: {best_params['win_rate']:.1f}%")
    print(f"Target hits: {best_params['target_hit_trades']}, Timeouts: {best_params['timeout_trades']}")
    print(f"Average duration: {best_params['avg_duration_minutes']:.1f} minutes")
    
    # Run detailed backtest with best parameters
    trades_df, summary = backtester.backtest_strategy(
        start_date=start_date,
        end_date=end_date,
        ema_fast=int(best_params['ema_fast']),
        ema_slow=int(best_params['ema_slow']),
        target_points=target_points,
        timeout_minutes=timeout_minutes,
        stop_loss_points=stop_loss_points
    )
    
    # Display detailed results
    if len(trades_df) > 0:
        print(f"\nDetailed trade results:")
        print(trades_df.to_string())
        
        # Plot results (comment out if running without display)
        try:
            backtester.plot_results(trades_df, f"NFT50 EMA({best_params['ema_fast']}, {best_params['ema_slow']}) Strategy")
        except Exception as e:
            print(f"Could not display plots: {e}")
    else:
        print("No trades were generated. Check your date range and parameters.")
    
    return backtester, trades_df, grid_results

if __name__ == "__main__":
    # Run example with generated data
    # backtester, trades_df, grid_results = run_backtest_example()
    
    # =============================================================================
    # TRADING CONFIGURATION PARAMETERS
    # =============================================================================
    timeout_minutes = 5  # Maximum time to hold a position before force exit
    target_points = 2     # Profit target in points
    stop_loss_points = 2  # Stop loss in points below entry price
    ema_fast = 9         # Fast EMA period
    ema_slow = 15        # Slow EMA period
    # =============================================================================
    
    # Use your actual NFT50 data:
    print("=== NIFTY 50 EMA Crossover Backtesting ===")
    print(f"Configuration:")
    print(f"  - Timeout: {timeout_minutes} minutes")
    print(f"  - Target: {target_points} points")
    print(f"  - Stop Loss: {stop_loss_points} points")
    print(f"  - EMA Fast: {ema_fast}, EMA Slow: {ema_slow}")
    print("="*60)
    
    try:
        backtester, trades_df, grid_results = load_and_backtest_nft50(
            file_path="/Users/krishnayadav/Downloads/nifty_july_data/NIFTY 50_minute_data.csv",
            start_date="2025-01-01",  # Adjusted to a more likely date range
            end_date="2025-06-01",
            timeout_minutes=timeout_minutes,
            target_points=target_points,
            stop_loss_points=stop_loss_points,
            ema_fast=ema_fast,
            ema_slow=ema_slow
        )
        
        if backtester is not None and len(trades_df) > 0:
            # Calculate profit and loss statistics
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] <= 0]
            total_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
            net_profit = trades_df['profit'].sum()
            
            print("\n" + "="*60)
            print("FINAL RESULTS SUMMARY")
            print("="*60)
            print(f"📊 TRADE STATISTICS:")
            print(f"   Total trades executed: {len(trades_df)}")
            print(f"   Winning trades: {len(winning_trades)}")
            print(f"   Losing trades: {len(losing_trades)}")
            print(f"   Win rate: {(len(winning_trades) / len(trades_df) * 100):.1f}%")
            print(f"")
            print(f"💰 PROFIT & LOSS:")
            print(f"   Total PROFIT: {total_profit:.2f} points")
            print(f"   Total LOSS: {total_loss:.2f} points")
            print(f"   Net Profit: {net_profit:.2f} points")
            print(f"   Average profit per trade: {trades_df['profit'].mean():.2f} points")
            print(f"")
            print(f"📈 EXIT ANALYSIS:")
            print(f"   Target hits: {len(trades_df[trades_df['exit_reason'] == 'Target Hit'])}")
            print(f"   Stop losses: {len(trades_df[trades_df['exit_reason'] == 'Stop Loss'])}")
            print(f"   {timeout_minutes}-min timeouts: {len(trades_df[trades_df['exit_reason'].str.contains('Timeout')])}")
            print(f"   Average trade duration: {trades_df['duration_minutes'].mean():.1f} minutes")
            
            print(f"\n📋 FIRST 10 TRADES:")
            print(trades_df[['entry_time', 'entry_price', 'exit_price', 'profit', 'duration_minutes', 'exit_reason']].head(10).to_string())
            
        elif backtester is not None:
            print("Backtester loaded successfully but no trades were generated.")
            print("This might be due to:")
            print("1. Date range doesn't match your data")
            print("2. No EMA crossovers in the specified period")
            print("3. Insufficient data for the EMA calculations")
        else:
            print("Failed to load data. Please check the file path and format.")
            
    except Exception as e:
        print(f"Error during backtesting: {e}")
        print("Please check your file path and data format.")
        
        # Try to show some basic info about what went wrong
        try:
            import os
            if os.path.exists("/Users/krishnayadav/Downloads/nifty_july_data/NIFTY 50_minute_data.csv"):
                print("File exists, but there might be a format issue.")
                # Try to read just the header
                with open("/Users/krishnayadav/Downloads/nifty_july_data/NIFTY 50_minute_data.csv", 'r') as f:
                    header = f.readline().strip()
                    print(f"File header: {header}")
            else:
                print("File does not exist at the specified path.")
        except Exception as file_check_error:
            print(f"Could not check file: {file_check_error}")