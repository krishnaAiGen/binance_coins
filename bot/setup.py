#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Binance New Listing Trading Bot
Helps users configure the bot for first-time use
"""

import os
import json
import requests
from datetime import datetime

def create_env_file():
    """Create .env file with user input"""
    print("=== Binance New Listing Trading Bot Setup ===\n")
    
    # Get user inputs
    print("Please provide the following information:")
    api_key = input("Binance API Key: ").strip()
    api_secret = input("Binance API Secret: ").strip()
    slack_webhook = input("Slack Webhook URL: ").strip()
    
    print("\nTrading Parameters (press Enter for defaults):")
    balance_pct = input("Balance Percentage to use (default 90): ").strip() or "90"
    target_profit = input("Target Profit Percentage (default 15): ").strip() or "15"
    stop_loss = input("Stop Loss Percentage (default 2): ").strip() or "2"
    leverage = input("Leverage (default 3): ").strip() or "3"
    
    # Create .env content
    env_content = f"""# Binance API Configuration
BINANCE_API_KEY={api_key}
BINANCE_API_SECRET={api_secret}

# Slack Configuration
SLACK_WEBHOOK_URL={slack_webhook}

# Trading Parameters
BALANCE_PERCENTAGE={balance_pct}  # Percentage of balance to use for trading
TARGET_PROFIT_PERCENT={target_profit}  # Target profit percentage
STOP_LOSS_PERCENT={stop_loss}  # Stop loss percentage
LEVERAGE={leverage}  # Leverage for futures trading

# Bot Configuration
CHECK_INTERVAL=60  # Check for new listings every 60 seconds (at 02 seconds of each minute)
RETRY_DELAY=3  # Delay in seconds before retrying failed trades
MAX_RETRIES=3  # Maximum number of retry attempts
"""
    
    # Write to .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("\n✅ .env file created successfully!")
    except Exception as e:
        print(f"\n❌ Error creating .env file: {e}")
        return False
    
    return True

def test_binance_connection(api_key, api_secret):
    """Test Binance API connection"""
    try:
        from binance.client import Client
        client = Client(api_key, api_secret, tld='com')
        
        # Test connection
        account_info = client.get_account()
        print("✅ Binance API connection successful!")
        
        # Test futures connection
        futures_balance = client.futures_account_balance()
        print("✅ Binance Futures API connection successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Binance API connection failed: {e}")
        return False

def test_slack_webhook(webhook_url):
    """Test Slack webhook"""
    try:
        test_message = {
            "text": f"🤖 Bot Setup Test\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStatus: Configuration test successful!"
        }
        
        response = requests.post(webhook_url, json=test_message)
        
        if response.status_code == 200:
            print("✅ Slack webhook test successful!")
            return True
        else:
            print(f"❌ Slack webhook test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Slack webhook test failed: {e}")
        return False

def initialize_data_files():
    """Initialize the data files with current futures pairs"""
    try:
        print("\n📊 Initializing data files with current futures pairs...")
        
        # Get current futures pairs
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pairs = []
        for symbol in data['symbols']:
            if symbol['status'] == 'TRADING':
                pairs.append(symbol['symbol'])
        
        # Save to futures_pairs.json
        with open('futures_pairs.json', 'w') as f:
            json.dump(pairs, f, indent=2)
        
        # Initialize empty processed_pairs.json
        with open('processed_pairs.json', 'w') as f:
            json.dump([], f, indent=2)
        
        print(f"✅ Data files initialized with {len(pairs)} current futures pairs")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing data files: {e}")
        return False

def main():
    """Main setup function"""
    print("Starting bot setup...\n")
    
    # Check if .env already exists
    if os.path.exists('.env'):
        overwrite = input(".env file already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    # Create .env file
    if not create_env_file():
        return
    
    # Load the created .env file
    from dotenv import load_dotenv
    load_dotenv('.env')
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
    
    print("\n🔍 Testing connections...")
    
    # Test Binance connection
    binance_ok = test_binance_connection(api_key, api_secret)
    
    # Test Slack webhook
    slack_ok = test_slack_webhook(slack_webhook)
    
    if binance_ok and slack_ok:
        print("\n🎉 All connections successful!")
        
        # Initialize data files
        if initialize_data_files():
            print("\n✅ Setup completed successfully!")
            print("\nYou can now run the bot with:")
            print("python listing_monitor_bot.py")
        else:
            print("\n⚠️  Setup completed with warnings. You may need to run the bot once to initialize data files.")
    else:
        print("\n⚠️  Setup completed with errors. Please check your configuration before running the bot.")
    
    print("\n📖 Please read the README.md file for detailed usage instructions.")

if __name__ == "__main__":
    main() 