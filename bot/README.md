# Binance New Listing Trading Bot

An automated trading bot that monitors Binance futures for new coin listings and automatically places long trades with stop-loss and take-profit orders.

## Features

- **Real-time Monitoring**: Checks for new futures listings every minute at 02 seconds
- **Automatic Trading**: Places long positions on newly listed pairs
- **Risk Management**: Automatic stop-loss (2%) and take-profit (15%) orders
- **Slack Notifications**: Real-time alerts for new listings and trade executions
- **Retry Logic**: Attempts failed trades up to 3 times with 3-second delays
- **Balance Management**: Uses configurable percentage of account balance
- **Error Handling**: Comprehensive error handling with Slack notifications

## Requirements

- Python 3.8+
- Binance Futures Account with API access
- Slack Webhook URL for notifications

## Installation

1. **Clone the repository and navigate to the bot directory:**
   ```bash
   cd bot
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Copy `example.env` to `.env`: `cp example.env .env`
   - Edit the `.env` file with your actual credentials and settings

## Configuration

### Environment Variables (.env file)

```env
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Slack Configuration
SLACK_WEBHOOK_URL=your_slack_webhook_url_here

# Trading Parameters
BALANCE_PERCENTAGE=90  # Percentage of balance to use for trading
TARGET_PROFIT_PERCENT=15  # Target profit percentage
STOP_LOSS_PERCENT=2  # Stop loss percentage
LEVERAGE=3  # Leverage for futures trading

# Bot Configuration
CHECK_INTERVAL=60  # Check for new listings every 60 seconds
RETRY_DELAY=3  # Delay in seconds before retrying failed trades
MAX_RETRIES=3  # Maximum number of retry attempts
```

### Setting up Binance API

1. Log into your Binance account
2. Go to API Management
3. Create a new API key
4. Enable futures trading permissions
5. Add your IP address to the whitelist (recommended)
6. Copy the API Key and Secret Key to your `.env` file

### Setting up Slack Notifications

1. Create a Slack workspace or use an existing one
2. Create a new channel for bot notifications
3. Go to https://api.slack.com/messaging/webhooks
4. Create a new webhook for your channel
5. Copy the webhook URL to your `.env` file

## Usage

### Running the Bot

```bash
python listing_monitor_bot.py
```

The bot will:
1. Initialize and load existing data
2. Send a startup notification to Slack
3. Begin monitoring for new listings every minute at 02 seconds
4. Automatically place trades when new listings are detected
5. Send detailed notifications to Slack for all activities

### Stopping the Bot

Press `Ctrl+C` to stop the bot. It will send a shutdown notification to Slack.

## How It Works

### Monitoring Process

1. **Data Storage**: The bot maintains two JSON files:
   - `futures_pairs.json`: Current list of all futures pairs
   - `processed_pairs.json`: Pairs that have already been processed

2. **Detection Logic**: Every minute at 02 seconds, the bot:
   - Fetches current futures pairs from Binance API
   - Compares with stored data to find new listings
   - Filters out already processed pairs

3. **Trading Logic**: For each new listing:
   - Sends immediate Slack alert
   - Calculates position size based on balance percentage
   - Places market long order with configured leverage
   - Sets stop-loss order at 2% below entry price
   - Sets take-profit order at 15% above entry price
   - Sends detailed trade notification to Slack

### Risk Management

- **Position Sizing**: Uses configurable percentage of total balance
- **Leverage Control**: Configurable leverage (default 3x)
- **Stop Loss**: Automatic 2% stop-loss on all positions
- **Take Profit**: Automatic 15% take-profit target
- **Retry Logic**: Up to 3 attempts with 3-second delays for failed trades

### Error Handling

- Comprehensive exception handling throughout the application
- All errors are logged and sent to Slack
- Bot continues running even if individual operations fail
- Graceful shutdown with notifications

## File Structure

```
bot/
├── listing_monitor_bot.py    # Main bot application
├── futures_trader.py         # Futures trading logic
├── slack_notifier.py         # Slack notification system
├── list_binance_coin.py      # Original listing script
├── example.env               # Environment configuration example
├── .env                      # Environment configuration file (create from example.env)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── futures_pairs.json        # Current futures pairs (auto-generated)
└── processed_pairs.json      # Processed pairs log (auto-generated)
```

## Slack Notifications

The bot sends various types of notifications:

### Startup Notification
- Bot status and configuration
- Current number of pairs being monitored

### New Listing Alert
- Immediate notification when new listing is detected
- Symbol and timestamp

### Trade Notifications
- Successful trades with full details (entry, SL, TP, quantity, etc.)
- Failed trades with error messages
- Retry attempts and outcomes

### Error Notifications
- API errors, network issues, etc.
- Critical bot errors

### Shutdown Notification
- Bot stop confirmation
- Summary statistics

## Safety Considerations

1. **Test First**: Always test with small amounts on testnet or with minimal funds
2. **API Permissions**: Only enable necessary permissions on your Binance API key
3. **IP Whitelist**: Restrict API access to your server's IP address
4. **Monitor Closely**: Watch the bot's performance, especially initially
5. **Balance Limits**: Set appropriate balance percentage to limit risk
6. **Regular Updates**: Keep dependencies updated for security

## Troubleshooting

### Common Issues

1. **API Errors**: Check API key permissions and IP whitelist
2. **Network Issues**: Ensure stable internet connection
3. **Insufficient Balance**: Check futures account balance
4. **Precision Errors**: Bot automatically handles symbol precision
5. **Slack Not Working**: Verify webhook URL and channel permissions

### Logs and Debugging

- All operations are logged to console with timestamps
- Error details are sent to Slack
- Check `futures_pairs.json` and `processed_pairs.json` for data integrity

## Disclaimer

This bot is for educational and informational purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk and never trade more than you can afford to lose.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review console logs and Slack notifications
3. Verify configuration settings
4. Test with minimal amounts first

## License

This project is provided as-is without warranty. Use responsibly and in accordance with applicable laws and regulations. 