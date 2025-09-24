#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance New Listing Monitor Bot
Monitors for new futures listings and automatically places long trades
"""

import os
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from slack_notifier import SlackNotifier
from futures_trader import FuturesTrader

# Load environment variables
load_dotenv()

class ListingMonitorBot:
    def __init__(self):
        """Initialize the listing monitor bot"""
        try:
            # Initialize components
            self.slack = SlackNotifier()
            self.trader = FuturesTrader()
            
            # Load configuration from environment
            self.check_interval = int(os.getenv('CHECK_INTERVAL', 60))
            self.retry_delay = int(os.getenv('RETRY_DELAY', 2))
            self.max_retries = int(os.getenv('MAX_RETRIES', 3))
            
            # New trading configuration for multiple trades
            self.trade_interval = 15  # 15 seconds between trades
            self.total_trades = 8     # Total number of trades (8 trades over 2 minutes)
            # Trade amount will be calculated dynamically from balance and percentage
            
            # Data storage files
            self.futures_data_file = 'futures_pairs.json'
            self.processed_pairs_file = 'processed_pairs.json'
            
            # Initialize data storage
            self.current_futures_pairs = set()
            self.processed_pairs = set()
            
            # Load existing data
            self.load_existing_data()
            
            print("Listing monitor bot initialized successfully")
            print(f"Check interval: {self.check_interval} seconds")
            print(f"Retry delay: {self.retry_delay} seconds")
            print(f"Max retries: {self.max_retries}")
            print(f"Trade strategy: ONE trade per listing, up to {self.total_trades} attempts")
            print(f"Trade amount: Calculated from balance × balance_percentage")
            print(f"Attempt interval: {self.trade_interval} seconds between attempts")
            print(f"Strategy: Stop attempting once trade is successfully placed")
            
        except Exception as e:
            print(f"Error initializing listing monitor bot: {e}")
            raise

    def load_existing_data(self):
        """Load existing futures pairs and processed pairs data"""
        try:
            # Load current futures pairs
            if os.path.exists(self.futures_data_file):
                with open(self.futures_data_file, 'r') as f:
                    data = json.load(f)
                    self.current_futures_pairs = set(data)
                print(f"Loaded {len(self.current_futures_pairs)} existing futures pairs")
            else:
                print("No existing futures pairs data found, fetching initial data...")
                self.update_futures_pairs()
            
            # Load processed pairs (pairs we've already attempted to trade)
            if os.path.exists(self.processed_pairs_file):
                with open(self.processed_pairs_file, 'r') as f:
                    data = json.load(f)
                    self.processed_pairs = set(data)
                print(f"Loaded {len(self.processed_pairs)} processed pairs")
            
        except Exception as e:
            print(f"Error loading existing data: {e}")
            # Initialize with fresh data if loading fails
            self.update_futures_pairs()

    def save_data(self):
        """Save current data to files"""
        try:
            # Save current futures pairs
            with open(self.futures_data_file, 'w') as f:
                json.dump(list(self.current_futures_pairs), f, indent=2)
            
            # Save processed pairs
            with open(self.processed_pairs_file, 'w') as f:
                json.dump(list(self.processed_pairs), f, indent=2)
                
        except Exception as e:
            print(f"Error saving data: {e}")

    def get_futures_pairs(self):
        """Get all active futures trading pairs from Binance API"""
        try:
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pairs = []
            for symbol in data['symbols']:
                if symbol['status'] == 'TRADING':
                    pairs.append(symbol['symbol'])
            
            return set(pairs)
            
        except Exception as e:
            print(f"Error fetching futures pairs: {e}")
            self.slack.post_error_to_slack(f"Error fetching futures pairs: {e}")
            return None

    def update_futures_pairs(self):
        """Update the current futures pairs data"""
        try:
            new_pairs = self.get_futures_pairs()
            if new_pairs is not None:
                self.current_futures_pairs = new_pairs
                self.save_data()
                print(f"Updated futures pairs data: {len(self.current_futures_pairs)} pairs")
                return True
            return False
        except Exception as e:
            print(f"Error updating futures pairs: {e}")
            return False

    def detect_new_listings(self):
        """Detect new futures listings by comparing with stored data"""
        try:
            # Get current pairs from API
            current_api_pairs = self.get_futures_pairs()
            if current_api_pairs is None:
                return []
            
            # Find new pairs (in current API data but not in stored data)
            new_pairs = current_api_pairs - self.current_futures_pairs
            
            # Filter out pairs we've already processed
            truly_new_pairs = [pair for pair in new_pairs if pair not in self.processed_pairs]
            
            if truly_new_pairs:
                print(f"New listings detected: {truly_new_pairs}")
                
                # Update stored data
                self.current_futures_pairs = current_api_pairs
                self.save_data()
                
                return truly_new_pairs
            
            return []
            
        except Exception as e:
            print(f"Error detecting new listings: {e}")
            self.slack.post_error_to_slack(f"Error detecting new listings: {e}")
            return []

    def get_trade_amount(self):
        """Calculate trade amount based on available futures balance and balance percentage"""
        try:
            # Get available futures balance (excludes locked funds in positions)
            available_balance = self.trader.get_available_futures_balance()
            if available_balance <= 0:
                print("❌ No available USDT balance for trading")
                return 0
            
            # Get balance percentage from trader (which loads it from .env)
            balance_percentage = self.trader.balance_percentage
            
            # Calculate trade amount based on available balance
            trade_amount = available_balance * (balance_percentage / 100)
            
            print(f"💰 Available futures balance: ${available_balance:.2f} USDT")
            print(f"📊 Using {balance_percentage}% of available balance: ${trade_amount:.2f}")
            
            return trade_amount
            
        except Exception as e:
            print(f"❌ Error calculating trade amount: {e}")
            return 0

    def execute_trade_attempts(self, symbol):
        """Attempt to place ONE trade every 15 seconds for up to 8 attempts (2 minutes)"""
        try:
            # Calculate trade amount based on current balance
            trade_amount = self.get_trade_amount()
            if trade_amount <= 0:
                print("❌ Cannot proceed: Invalid trade amount")
                return False
            
            print(f"Starting trade attempts for {symbol}")
            print(f"Strategy: Attempt ONE trade of ${trade_amount:.2f} every {self.trade_interval} seconds")
            print(f"Max attempts: {self.total_trades} (will stop once successful)")
            
            failed_attempts = []
            successful_trade = None
            trade_placed = False
            
            for attempt_num in range(1, self.total_trades + 1):
                try:
                    print(f"\n--- Attempt #{attempt_num}/{self.total_trades} for {symbol} ---")
                    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Place fixed amount trade without order splitting
                    result = self.trader.place_fixed_amount_trade(symbol, trade_amount)
                    
                    if result['success']:
                        print(f"✅ Trade SUCCESSFUL on attempt #{attempt_num}!")
                        print(f"🎯 STOPPING further attempts - trade placed successfully")
                        
                        successful_trade = {
                            'attempt_number': attempt_num,
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'result': result
                        }
                        trade_placed = True
                        break  # EXIT LOOP - No more attempts needed
                    else:
                        print(f"❌ Attempt #{attempt_num} failed: {result.get('error', 'Unknown error')}")
                        failed_attempts.append({
                            'attempt_number': attempt_num,
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'error': result.get('error', 'Unknown error')
                        })
                        
                        # Wait 15 seconds before next attempt (except for the last attempt)
                        if attempt_num < self.total_trades:
                            print(f"⏰ Waiting {self.trade_interval} seconds before next attempt...")
                            time.sleep(self.trade_interval)
                    
                except Exception as e:
                    error_msg = f"Exception during attempt #{attempt_num} for {symbol}: {e}"
                    print(f"❌ {error_msg}")
                    failed_attempts.append({
                        'attempt_number': attempt_num,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'error': str(e)
                    })
                    
                    # Continue with next attempt even if current one failed
                    if attempt_num < self.total_trades:
                        print(f"⏰ Waiting {self.trade_interval} seconds before next attempt...")
                        time.sleep(self.trade_interval)
            
            # Summary
            print(f"\n🎯 TRADE ATTEMPTS SUMMARY for {symbol}:")
            print(f"Total attempts made: {len(failed_attempts) + (1 if trade_placed else 0)}")
            print(f"Trade placed: {'✅ YES' if trade_placed else '❌ NO'}")
            print(f"Failed attempts: {len(failed_attempts)}")
            
            if trade_placed:
                print(f"✅ SUCCESS: Trade placed on attempt #{successful_trade['attempt_number']}")
                print(f"💰 Trade amount: ${trade_amount:.2f}")
            else:
                print(f"❌ FAILED: All {self.total_trades} attempts failed")
            
            # Send summary notification to Slack
            summary_info = {
                'symbol': symbol,
                'total_attempts': len(failed_attempts) + (1 if trade_placed else 0),
                'max_attempts': self.total_trades,
                'trade_placed': trade_placed,
                'successful_attempt': successful_trade['attempt_number'] if trade_placed else None,
                'failed_attempts': len(failed_attempts),
                'trade_amount': trade_amount,
                'failed_attempt_numbers': [a['attempt_number'] for a in failed_attempts],
                'strategy': 'Single trade with multiple attempts'
            }
            
            self.slack.post_trade_attempts_summary(summary_info)
            
            return trade_placed  # Return True only if trade was successfully placed
            
        except Exception as e:
            error_msg = f"Critical error during trade attempts for {symbol}: {e}"
            print(error_msg)
            self.slack.post_error_to_slack(error_msg)
            return False

    def process_new_listing(self, symbol):
        """Process a new listing by waiting 15 seconds then attempting to place one trade"""
        try:
            # Send new listing alert to Slack
            self.slack.post_new_listing_alert(symbol)
            
            # Mark as processed to avoid duplicate attempts
            self.processed_pairs.add(symbol)
            self.save_data()
            
            # Wait 15 seconds after listing detection before starting trade attempts
            print(f"⏰ Waiting 15 seconds before starting trade attempts for {symbol}...")
            time.sleep(10)
            
            # Execute trade attempts (one trade, multiple attempts until success)
            success = self.execute_trade_attempts(symbol)
            
            if success:
                print(f"Successfully processed new listing: {symbol}")
            else:
                print(f"Failed to process new listing: {symbol}")
                
            return success
            
        except Exception as e:
            error_msg = f"Error processing new listing {symbol}: {e}"
            print(error_msg)
            self.slack.post_error_to_slack(error_msg)
            return False

    def wait_for_target_time(self):
        """Wait until the next target time (02 seconds of each minute)"""
        try:
            while True:
                now = datetime.now()
                current_second = now.second
                
                # Target is 02 seconds of each minute
                if current_second == 2:
                    break
                
                # Calculate sleep time to reach target second
                if current_second < 2:
                    sleep_time = 2 - current_second
                else:
                    sleep_time = 62 - current_second  # Wait for next minute
                
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Error in wait_for_target_time: {e}")

    def run(self):
        """Main bot loop"""
        print("Starting Binance New Listing Monitor Bot...")
        print("Bot will check for new listings at 02 seconds of every minute")
        
        # Send startup notification
        startup_message = {
            "🤖 BOT STARTED": "✅ MONITORING ACTIVE",
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Status": "Monitoring for new futures listings",
            "Check Interval": f"Every minute at 02 seconds",
            "Current Pairs": len(self.current_futures_pairs),
            "Processed Pairs": len(self.processed_pairs)
        }
        self.slack.post_to_slack(startup_message)
        
        try:
            while True:
                # Wait for target time (02 seconds of each minute)
                self.wait_for_target_time()
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{current_time}] Checking for new listings...")
                
                # Detect new listings
                new_listings = self.detect_new_listings()
                new_listings = ['BLESSUSDT']
                
                if new_listings:
                    print(f"Found {len(new_listings)} new listings: {new_listings}")
                    
                    # Process each new listing
                    for symbol in new_listings:
                        print(f"Processing new listing: {symbol}")
                        self.process_new_listing(symbol)
                        
                        # Small delay between processing multiple listings
                        if len(new_listings) > 1:
                            time.sleep(1)
                else:
                    print("No new listings detected")
                
                # Sleep until next check (approximately 60 seconds from now)
                time.sleep(58)  # Sleep for 58 seconds, then wait_for_target_time will handle the precise timing
                
        except KeyboardInterrupt:
            print("\nBot stopped by user")
            
            # Send shutdown notification
            shutdown_message = {
                "🛑 BOT STOPPED": "⚠️ MONITORING INACTIVE",
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Status": "Bot manually stopped",
                "Total Processed": len(self.processed_pairs)
            }
            self.slack.post_to_slack(shutdown_message)
            
        except Exception as e:
            error_msg = f"Critical error in bot main loop: {e}"
            print(error_msg)
            self.slack.post_error_to_slack(error_msg)
            raise

def update_futures_pairs_data():
    """Download and update futures_pairs.json with current Binance data"""
    try:
        print("🔄 Updating futures pairs data...")
        
        # Get current futures pairs from API
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pairs = []
        for symbol in data['symbols']:
            if symbol['status'] == 'TRADING':
                pairs.append(symbol['symbol'])
        
        # Backup existing file if it exists
        if os.path.exists('futures_pairs.json'):
            backup_name = f"futures_pairs.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename('futures_pairs.json', backup_name)
            print(f"📁 Backed up existing file as {backup_name}")
        
        # Save new data
        with open('futures_pairs.json', 'w') as f:
            json.dump(pairs, f, indent=2)
        
        print(f"✅ Updated futures_pairs.json with {len(pairs)} current pairs")
        return True
        
    except Exception as e:
        print(f"❌ Error updating futures pairs data: {e}")
        return False

if __name__ == "__main__":
    try:
        # First, update futures pairs data to ensure we have current listings
        print("🚀 === Binance New Listing Monitor Bot ===")
        update_success = update_futures_pairs_data()
        
        if not update_success:
            print("⚠️  Warning: Could not update futures pairs data, using existing file...")
        
        print("🤖 Starting bot...")
        bot = ListingMonitorBot()
        bot.run()
    except Exception as e:
        print(f"Failed to start bot: {e}")
        # Try to send error notification if possible
        try:
            slack = SlackNotifier()
            slack.post_error_to_slack(f"Failed to start listing monitor bot: {e}")
        except:
            pass 