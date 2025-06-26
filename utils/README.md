# Binance Coin Listing Analyzer

This directory contains utilities to analyze Binance trading pairs and their listing dates.

## Files

### `list_binance_coin.py` (Original)
Simple script that fetches current active trading pairs without dates.

### `list_binance_coin_with_dates.py` (Enhanced)
Advanced script that fetches all trading pairs with their listing dates by analyzing the first available candlestick data.

## Features

- 📊 Fetches all active spot and futures pairs
- 📅 Determines exact listing dates using first available kline data
- 🔄 Sorts pairs chronologically by listing date
- 💾 Saves data in both JSON and CSV formats
- 📈 Shows detailed progress and statistics
- ⚡ Handles API rate limits with delays
- 🛡️ Robust error handling

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Binance API credentials in `.env` file:
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

Note: The script will look for `.env` in multiple locations:
- Current directory
- Parent directory (`../.env`)
- Bot directory (`../bot/.env`)

## Usage

### Basic Listing (No Dates)
```bash
python list_binance_coin.py
```
Output: `spot.json`, `futures.json`

### Enhanced Listing with Dates
```bash
python list_binance_coin_with_dates.py
```
Output: 
- `spot_with_dates.json` & `spot_with_dates.csv`
- `futures_with_dates.json` & `futures_with_dates.csv`

## Output Format

### JSON Structure
```json
[
  {
    "symbol": "BTCUSDT",
    "baseAsset": "BTC",
    "quoteAsset": "USDT",
    "contractType": "PERPETUAL",  // futures only
    "listing_date": "2019-09-08 00:00:00"
  }
]
```

### CSV Columns
- `symbol`: Trading pair symbol
- `baseAsset`: Base asset (e.g., BTC)
- `quoteAsset`: Quote asset (e.g., USDT)
- `contractType`: Contract type (futures only)
- `listing_date`: First available trading date

## Features

### Chronological Sorting
Pairs are sorted by listing date (oldest first), making it easy to:
- Identify the oldest and newest listings
- Analyze listing patterns over time
- Use for backtesting strategies

### Progress Tracking
- Real-time progress display
- Progress saves every 100 pairs
- Detailed statistics and summaries

### Error Handling
- Graceful handling of API errors
- Continues processing even if some pairs fail
- Clear error messages and warnings

## API Requirements

- Binance API credentials (for historical data access)
- No special permissions required (read-only access)
- Respects rate limits with automatic delays

## Performance

- Processes ~2000+ pairs (spot + futures)
- Takes approximately 5-10 minutes depending on API response times
- Uses minimal API calls (1 per pair for listing date)
- Progress is saved incrementally

## Use Cases

1. **Trading Strategy Development**: Understand when pairs were listed
2. **Backtesting**: Use chronological data for historical analysis
3. **Market Research**: Analyze Binance's listing patterns
4. **Portfolio Management**: Track available trading pairs over time 