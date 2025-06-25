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
            self.retry_delay = int(os.getenv('RETRY_DELAY', 3))
            self.max_retries = int(os.getenv('MAX_RETRIES', 3))
            
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

    def attempt_trade_with_retry(self, symbol):
        """Attempt to place a trade with retry logic"""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"Attempt {attempt}/{self.max_retries} to trade {symbol}")
                
                # Attempt to place long trade
                result = self.trader.place_long_trade(symbol)
                
                if result['success']:
                    print(f"Successfully placed trade for {symbol} on attempt {attempt}")
                    if attempt > 1:
                        self.slack.post_retry_notification(symbol, attempt, success=True)
                    return True
                else:
                    print(f"Trade failed for {symbol} on attempt {attempt}: {result.get('error', 'Unknown error')}")
                    
                    if attempt < self.max_retries:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        self.slack.post_retry_notification(symbol, attempt, success=False)
                        time.sleep(self.retry_delay)
                    
            except Exception as e:
                error_msg = f"Exception during trade attempt {attempt} for {symbol}: {e}"
                print(error_msg)
                
                if attempt < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.slack.post_error_to_slack(error_msg)
        
        print(f"All {self.max_retries} attempts failed for {symbol}")
        return False

    def process_new_listing(self, symbol):
        """Process a new listing by attempting to trade it"""
        try:
            # Send new listing alert to Slack
            self.slack.post_new_listing_alert(symbol)
            
            # Mark as processed to avoid duplicate attempts
            self.processed_pairs.add(symbol)
            self.save_data()
            
            # Attempt to place trade with retry logic
            success = self.attempt_trade_with_retry(symbol)
            
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

if __name__ == "__main__":
    try:
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