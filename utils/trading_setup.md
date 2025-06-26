# Manual Long Position Trading Setup

## Prerequisites

1. **Install Dependencies**: Make sure you have the required Python packages:
   ```bash
   pip install python-binance python-dotenv
   ```

2. **Environment Variables**: Create a `.env` file in the project root with your Binance API credentials:
   ```
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_API_SECRET=your_binance_api_secret_here
   LEVERAGE=3
   TARGET_PROFIT_PERCENT=15
   STOP_LOSS_PERCENT=2
   ```

3. **Binance API Setup**:
   - Go to Binance > Account > API Management
   - Create a new API key
   - Enable Futures Trading permissions
   - Add your server IP to the whitelist (if required)

## Usage

Run the script from the utils directory:
```bash
cd utils
python test_long.py
```

The script will:
1. Show your current futures balance
2. Ask for the coin symbol (e.g., BTC, ETH)
3. Ask for the trade amount in USDT
4. Show a confirmation with all trade details
5. Execute the trade with automatic stop loss and take profit

## Example

```
Current Futures Balance: 1000.0 USDT

Enter coin symbol (e.g., BTCUSDT, ETHUSDT): BTC
Enter trade amount in USDT: 100

TRADE CONFIRMATION:
Symbol: BTCUSDT
Trade Amount: 100.0 USDT
Leverage: 3x
Stop Loss: 2%
Take Profit: 15%

Do you want to proceed with this trade? (y/N): y
```

## Safety Features

- Balance validation (checks if you have sufficient funds)
- Symbol validation (ensures the trading pair exists)
- Automatic stop loss and take profit orders
- Trade confirmation before execution
- Detailed trade summary after execution

## Customization

You can modify the following parameters in your `.env` file:
- `LEVERAGE`: Leverage multiplier (default: 3x)
- `TARGET_PROFIT_PERCENT`: Take profit percentage (default: 15%)
- `STOP_LOSS_PERCENT`: Stop loss percentage (default: 2%) 