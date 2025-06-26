# Binance Futures New Listing Backtesting System

This directory contains backtesting tools to analyze the performance of the new listing trading strategy with historical data and simulations.

## Files Overview

### 1. `futures_listing_backtest.py`
**Full Historical Backtesting System**
- Fetches real historical data from Binance API
- Analyzes actual new listings from the past 90 days
- Requires Binance API credentials
- Provides comprehensive analysis with real market data

### 2. `manual_backtest.py` 
**Simulation-Based Backtesting**
- Generates realistic synthetic price data
- Simulates various market scenarios for new listings
- No API credentials required
- Quick testing and strategy validation

### 3. `requirements.txt`
**Dependencies**
- All required Python packages for backtesting

## Strategy Parameters

Both backtesting systems test the following strategy:
- **Initial Balance**: $1,000
- **Stop Loss**: 2%
- **Take Profit**: 15%
- **Leverage**: 1x (No leverage)
- **Position Size**: 90% of available balance
- **Entry Price**: Very first price of the coin (opening price of first candle)
- **Time Frame**: 24 hours maximum hold time
- **Interval**: 5-minute candles

## Quick Start

### Option 1: Manual Simulation (Recommended for Testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Run manual backtest
python manual_backtest.py
```

This will:
- Simulate 30 new listing trades
- Generate realistic price scenarios
- Create a detailed CSV report
- Show comprehensive statistics

### Option 2: Real Historical Data

```bash
# Set your Binance API credentials
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# Run historical backtest
python futures_listing_backtest.py
```

This will:
- Find actual new listings from the past 90 days
- Fetch real 5-minute price data
- Analyze actual market performance
- Generate detailed CSV report

## Output CSV Columns

The backtesting generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| **Symbol** | Trading pair symbol |
| **Scenario** | Market scenario (manual backtest only) |
| **Entry Time** | When the trade was entered |
| **Entry Price** | Price at which position was opened |
| **Exit Time** | When the trade was closed |
| **Exit Price** | Price at which position was closed |
| **Exit Reason** | Stop Loss / Take Profit / 24h Timeout |
| **Time to Exit (Minutes)** | Duration of the trade |
| **Stop Loss Hit Time** | When SL was triggered (if applicable) |
| **Take Profit Hit Time** | When TP was triggered (if applicable) |
| **PnL ($)** | Profit/Loss in dollars |
| **PnL (%)** | Profit/Loss percentage |
| **Highest Price 24h** | Peak price within 24 hours |
| **Lowest Price 24h** | Lowest price within 24 hours |
| **Highest Gain (%)** | Maximum unrealized gain |
| **Lowest Gain (%)** | Maximum unrealized loss |
| **Max Profit (%)** | Peak profit during trade |
| **Max Drawdown (%)** | Worst drawdown during trade |
| **Balance After** | Account balance after trade |
| **Cumulative Return (%)** | Total return from start |

## Market Scenarios (Manual Backtest)

The manual backtesting simulates realistic new listing scenarios:

1. **pump_dump**: Quick initial pump followed by dump
2. **steady_decline**: Gradual decline over 24 hours
3. **volatile_up**: High volatility with upward bias
4. **volatile_down**: High volatility with downward bias
5. **sideways**: Mostly sideways movement with volatility

## Sample Results Analysis

The system provides comprehensive statistics including:

### Performance Metrics
- Initial vs Final Balance
- Total Return ($) and (%)
- Win Rate and Average Win/Loss
- Largest Win/Loss
- Average Time to Exit

### Exit Reason Breakdown
- Percentage of trades hitting Stop Loss
- Percentage of trades hitting Take Profit
- Percentage of trades timing out at 24h

### Scenario Performance (Manual)
- Performance breakdown by market scenario
- Identifies which scenarios are most profitable

## Example Output

```
📊 BACKTEST RESULTS SUMMARY
================================================================================
Initial Balance: $1,000.00
Final Balance: $1,247.50
Total Return: $247.50
Total Return %: 24.75%

📈 TRADE ANALYSIS:
Total Trades: 30
Winning Trades: 18 (60.0%)
Losing Trades: 12 (40.0%)
Average Win: $45.30
Average Loss: -$18.20
Largest Win: $135.50
Largest Loss: -$21.40
Average Time to Exit: 287.3 minutes

🎯 EXIT REASONS:
Take Profit: 15 (50.0%)
Stop Loss: 10 (33.3%)
24h Timeout: 5 (16.7%)
```

## Customization

### Modify Trading Parameters

Edit the parameters in either script:

```python
self.initial_balance = 1000.0      # Starting balance
self.stop_loss_pct = 2.0           # Stop loss percentage
self.take_profit_pct = 15.0        # Take profit percentage
self.leverage = 1                  # No leverage (1x)
self.balance_percentage = 90       # % of balance to use
```

### Adjust Number of Trades

For manual backtest:
```python
results_df = backtester.run_backtest(num_trades=50)  # Test 50 trades
```

For historical backtest:
```python
results_df = backtester.run_backtest(days_back=180)  # Look back 180 days
```

## Interpreting Results

### Key Metrics to Watch

1. **Win Rate**: Should be >50% for profitable strategy
2. **Risk/Reward Ratio**: Average win should be >2x average loss
3. **Time to Exit**: Shorter times indicate quicker decisions
4. **Max Drawdown**: Monitor risk exposure
5. **Cumulative Return**: Overall strategy performance

### Red Flags

- Win rate <40%
- Average loss > Average win
- High percentage of 24h timeouts
- Consistently negative returns across scenarios

## Limitations

### Manual Backtest
- Synthetic data may not reflect real market conditions
- Scenarios are simplified representations
- No liquidity or slippage considerations

### Historical Backtest
- Limited to available historical data
- Past performance doesn't guarantee future results
- API rate limits may affect data collection
- Market conditions change over time

## Tips for Better Backtesting

1. **Run Multiple Tests**: Test with different parameters
2. **Analyze Scenarios**: Understand which market conditions work best
3. **Consider Slippage**: Real trading has execution delays
4. **Factor in Fees**: Add trading fees to calculations
5. **Test Different Timeframes**: Try different analysis periods
6. **Validate with Paper Trading**: Test with real market conditions

## Next Steps

After backtesting:

1. **Analyze Results**: Understand the strategy's strengths and weaknesses
2. **Optimize Parameters**: Adjust SL/TP levels based on results
3. **Paper Trade**: Test with real market data before live trading
4. **Risk Management**: Set position sizing based on backtest results
5. **Monitor Performance**: Compare live results with backtest expectations

## Disclaimer

Backtesting results are not indicative of future performance. Past results do not guarantee future profits. Cryptocurrency trading involves substantial risk of loss. Always trade responsibly and never risk more than you can afford to lose. 