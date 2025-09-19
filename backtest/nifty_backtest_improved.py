#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved NIFTY 50 EMA Crossover Strategy
Enhanced with better risk management and multiple exit conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class ImprovedNiftyBacktester:
    def __init__(self, data, date_column='timestamp'):
        """Initialize the improved backtester"""
        self.data = data.copy()
        
        # Handle different date column names
        if date_column in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data[date_column])
        elif 'date' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['date'])
        elif 'Date' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['Date'])
        else:
            first_col = self.data.columns[0]
            self.data['timestamp'] = pd.to_datetime(self.data[first_col])
        
        # Ensure required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in self.data.columns:
                if col.capitalize() in self.data.columns:
                    self.data[col] = self.data[col.capitalize()]
                elif col.upper() in self.data.columns:
                    self.data[col] = self.data[col.upper()]
        
        if 'volume' not in self.data.columns:
            self.data['volume'] = 1000
        
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Improved Backtester initialized with {len(self.data)} rows")
    
    def calculate_indicators(self, df, ema_fast, ema_slow, atr_period=14):
        """Calculate technical indicators including ATR for dynamic stops"""
        # EMAs
        df[f'EMA_{ema_fast}'] = df['close'].ewm(span=ema_fast).mean()
        df[f'EMA_{ema_slow}'] = df['close'].ewm(span=ema_slow).mean()
        
        # ATR for dynamic stop loss
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=atr_period).mean()
        
        # EMA crossovers
        df['EMA_Cross'] = 0
        df.loc[(df[f'EMA_{ema_fast}'] > df[f'EMA_{ema_slow}']) & 
               (df[f'EMA_{ema_fast}'].shift(1) <= df[f'EMA_{ema_slow}'].shift(1)), 'EMA_Cross'] = 1
        
        # Market trend filter
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['Trend_Up'] = df['close'] > df['EMA_50']
        
        # Volatility filter
        df['Price_Change'] = abs(df['close'].pct_change())
        df['Vol_Filter'] = df['Price_Change'] < df['Price_Change'].rolling(20).quantile(0.8)
        
        return df
    
    def improved_backtest(self, start_date, end_date, ema_fast=9, ema_slow=15,
                         target_points=2, stop_loss_atr_mult=1.5, timeout_minutes=10,
                         use_trend_filter=True, use_vol_filter=True, 
                         trailing_stop=False, risk_reward_ratio=1.0):
        """
        Improved backtesting with multiple enhancements:
        - ATR-based dynamic stop loss
        - Trend filtering
        - Volatility filtering  
        - Trailing stop option
        - Risk/reward optimization
        """
        
        # Filter data by date range
        mask = (self.data['timestamp'] >= start_date) & (self.data['timestamp'] <= end_date)
        df = self.data[mask].copy().reset_index(drop=True)
        
        if len(df) < 100:
            return pd.DataFrame(), {}
        
        # Calculate indicators
        df = self.calculate_indicators(df, ema_fast, ema_slow)
        
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        entry_idx = 0
        stop_loss_price = 0
        trailing_stop_price = 0
        
        for i in range(50, len(df)):  # Start after indicators are calculated
            current_row = df.iloc[i]
            
            # Entry logic with filters
            if not in_position and i > 0:
                prev_row = df.iloc[i-1]
                
                # Basic EMA cross signal
                ema_signal = prev_row['EMA_Cross'] == 1
                
                # Apply filters
                trend_ok = not use_trend_filter or current_row['Trend_Up']
                vol_ok = not use_vol_filter or current_row['Vol_Filter']
                
                if ema_signal and trend_ok and vol_ok:
                    # Enter position
                    in_position = True
                    entry_price = current_row['open']
                    entry_time = current_row['timestamp']
                    entry_idx = i
                    
                    # Calculate dynamic stop loss using ATR
                    if pd.notna(current_row['ATR']):
                        atr_stop = entry_price - (current_row['ATR'] * stop_loss_atr_mult)
                        # Use minimum of ATR stop and fixed point stop
                        fixed_stop = entry_price - (target_points * risk_reward_ratio)
                        stop_loss_price = max(atr_stop, fixed_stop)
                    else:
                        stop_loss_price = entry_price - (target_points * risk_reward_ratio)
                    
                    trailing_stop_price = stop_loss_price
                    
                    # Calculate target price
                    target_price = entry_price + target_points
            
            # Exit logic
            if in_position:
                duration_minutes = (current_row['timestamp'] - entry_time).total_seconds() / 60
                
                # Update trailing stop
                if trailing_stop:
                    new_trailing_stop = current_row['low'] - (current_row['ATR'] * stop_loss_atr_mult if pd.notna(current_row['ATR']) else target_points)
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop)
                    effective_stop = trailing_stop_price
                else:
                    effective_stop = stop_loss_price
                
                exit_price = None
                exit_reason = None
                
                # Check exit conditions in order of priority
                
                # 1. Stop loss hit
                if current_row['low'] <= effective_stop:
                    exit_price = effective_stop
                    exit_reason = "Stop Loss"
                
                # 2. Target hit
                elif current_row['high'] >= entry_price + target_points:
                    exit_price = entry_price + target_points
                    exit_reason = "Target Hit"
                
                # 3. Timeout (last resort)
                elif duration_minutes >= timeout_minutes:
                    exit_price = current_row['close']
                    exit_reason = f"{timeout_minutes}min Timeout"
                
                # Execute exit
                if exit_price is not None:
                    profit = exit_price - entry_price
                    
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current_row['timestamp'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'duration_minutes': duration_minutes,
                        'exit_reason': exit_reason,
                        'stop_loss_price': effective_stop,
                        'atr_at_entry': current_row['ATR'] if pd.notna(current_row['ATR']) else 0,
                        'trend_filter': current_row['Trend_Up'],
                        'vol_filter': current_row['Vol_Filter']
                    }
                    trades.append(trade)
                    in_position = False
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate summary statistics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] <= 0]
            
            summary = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades_df)) * 100,
                'total_profit': winning_trades['profit'].sum() if len(winning_trades) > 0 else 0,
                'total_loss': abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0,
                'net_profit': trades_df['profit'].sum(),
                'avg_win': winning_trades['profit'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': abs(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 0,
                'profit_factor': (winning_trades['profit'].sum() / abs(losing_trades['profit'].sum())) if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf'),
                'avg_duration': trades_df['duration_minutes'].mean(),
                'max_win': trades_df['profit'].max(),
                'max_loss': trades_df['profit'].min(),
                'target_hits': len(trades_df[trades_df['exit_reason'] == 'Target Hit']),
                'stop_losses': len(trades_df[trades_df['exit_reason'] == 'Stop Loss']),
                'timeouts': len(trades_df[trades_df['exit_reason'].str.contains('Timeout')]),
            }
        else:
            summary = {k: 0 for k in ['total_trades', 'winning_trades', 'losing_trades', 'win_rate', 
                                     'total_profit', 'total_loss', 'net_profit', 'avg_win', 'avg_loss',
                                     'profit_factor', 'avg_duration', 'max_win', 'max_loss',
                                     'target_hits', 'stop_losses', 'timeouts']}
        
        return trades_df, summary

def compare_strategies():
    """Compare different strategy configurations"""
    
    # Load data (using the same path as before)
    try:
        df = pd.read_csv("/Users/krishnayadav/Downloads/nifty_july_data/NIFTY 50_minute_data.csv")
        backtester = ImprovedNiftyBacktester(df)
        
        print("🔄 STRATEGY COMPARISON")
        print("="*60)
        
        strategies = [
            {
                'name': 'Original Strategy',
                'params': {
                    'target_points': 2,
                    'stop_loss_atr_mult': 0,  # No ATR stop
                    'timeout_minutes': 5,
                    'use_trend_filter': False,
                    'use_vol_filter': False,
                    'trailing_stop': False,
                    'risk_reward_ratio': 10  # Very loose stop loss
                }
            },
            {
                'name': 'Fixed Risk/Reward 1:1',
                'params': {
                    'target_points': 2,
                    'stop_loss_atr_mult': 0,
                    'timeout_minutes': 10,
                    'use_trend_filter': False,
                    'use_vol_filter': False,
                    'trailing_stop': False,
                    'risk_reward_ratio': 1.0  # 1:1 risk reward
                }
            },
            {
                'name': 'ATR-Based Stops',
                'params': {
                    'target_points': 2,
                    'stop_loss_atr_mult': 1.5,
                    'timeout_minutes': 10,
                    'use_trend_filter': False,
                    'use_vol_filter': False,
                    'trailing_stop': False,
                    'risk_reward_ratio': 1.0
                }
            },
            {
                'name': 'With Trend Filter',
                'params': {
                    'target_points': 2,
                    'stop_loss_atr_mult': 1.5,
                    'timeout_minutes': 10,
                    'use_trend_filter': True,
                    'use_vol_filter': False,
                    'trailing_stop': False,
                    'risk_reward_ratio': 1.0
                }
            },
            {
                'name': 'Full Enhancement',
                'params': {
                    'target_points': 2,
                    'stop_loss_atr_mult': 1.5,
                    'timeout_minutes': 10,
                    'use_trend_filter': True,
                    'use_vol_filter': True,
                    'trailing_stop': True,
                    'risk_reward_ratio': 1.0
                }
            }
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\n📊 Testing: {strategy['name']}")
            
            trades_df, summary = backtester.improved_backtest(
                start_date="2025-01-01",
                end_date="2025-02-07",
                **strategy['params']
            )
            
            summary['strategy_name'] = strategy['name']
            results.append(summary)
            
            print(f"   Trades: {summary['total_trades']}")
            print(f"   Win Rate: {summary['win_rate']:.1f}%")
            print(f"   Net Profit: {summary['net_profit']:.2f} points")
            print(f"   Profit Factor: {summary['profit_factor']:.2f}")
            print(f"   Avg Win: {summary['avg_win']:.2f}, Avg Loss: {summary['avg_loss']:.2f}")
        
        # Create comparison table
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*100)
        print("📈 STRATEGY COMPARISON RESULTS")
        print("="*100)
        
        comparison_cols = ['strategy_name', 'total_trades', 'win_rate', 'net_profit', 
                          'profit_factor', 'avg_win', 'avg_loss', 'target_hits', 
                          'stop_losses', 'timeouts']
        
        print(results_df[comparison_cols].to_string(index=False))
        
        # Find best strategy
        best_strategy = results_df.loc[results_df['net_profit'].idxmax()]
        print(f"\n🏆 BEST STRATEGY: {best_strategy['strategy_name']}")
        print(f"   Net Profit: {best_strategy['net_profit']:.2f} points")
        print(f"   Win Rate: {best_strategy['win_rate']:.1f}%")
        print(f"   Profit Factor: {best_strategy['profit_factor']:.2f}")
        
        return results_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    results = compare_strategies() 