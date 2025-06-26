#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Long Position Trading Script
Allows specifying trade amount and coin symbol for long positions
"""

import sys
import os
import math
from datetime import datetime

# Add the bot directory to the Python path to import FuturesTrader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bot'))

from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# TRADING CONFIGURATION - MODIFY THESE VARIABLES
# =============================================================================
COIN_NAME = "SAHARA"  # Change this to your desired coin (e.g., "BTC", "ETH", "SOL")
TRADE_AMOUNT = 1000.0  # Change this to your desired trade amount in USDT
# =============================================================================

class ManualFuturesTrader:
    def __init__(self):
        """Initialize the Manual Futures Trader with API credentials from environment variables"""
        try:
            # Get API credentials from environment variables
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in environment variables")
            
            # Initialize the Binance client
            self.client = Client(api_key, api_secret, tld='com')
            
            # Load trading parameters from environment (with defaults)
            self.target_profit_percent = float(os.getenv('TARGET_PROFIT_PERCENT', 15))
            self.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', 2))
            self.leverage = int(os.getenv('LEVERAGE', 3))
            
            print("Manual Futures trader initialized successfully")
            
        except Exception as e:
            print(f"Error initializing futures trader: {e}")
            raise

    def get_futures_balance(self):
        """Get USDT balance from futures account"""
        try:
            balance_info = self.client.futures_account_balance()
            for asset in balance_info:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0
        except Exception as e:
            print(f"Error getting futures balance: {e}")
            return 0.0

    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None

    def get_symbol_precision(self, symbol):
        """Get quantity precision for a symbol"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            if step_size == 1.0:
                                return 0
                            elif step_size == 0.1:
                                return 1
                            elif step_size == 0.01:
                                return 2
                            elif step_size == 0.001:
                                return 3
                            else:
                                # Calculate precision from step size
                                return len(str(step_size).split('.')[-1].rstrip('0'))
            return 3  # Default precision
        except Exception as e:
            print(f"Error getting symbol precision for {symbol}: {e}")
            return 3  # Default precision

    def calculate_quantity_from_amount(self, symbol, trade_amount, current_price):
        """Calculate quantity based on specified trade amount and current price"""
        try:
            # Calculate quantity with leverage
            quantity = (trade_amount * self.leverage) / current_price
            
            # Get symbol precision and round quantity
            precision = self.get_symbol_precision(symbol)
            rounded_quantity = round(quantity, precision)
            
            print(f"Trade amount: {trade_amount} USDT")
            print(f"Current price: {current_price}")
            print(f"Leverage: {self.leverage}x")
            print(f"Calculated quantity: {rounded_quantity}")
            return rounded_quantity
            
        except Exception as e:
            print(f"Error calculating quantity: {e}")
            return None

    def get_price_precision(self, price):
        """Get appropriate price precision based on price value"""
        if price <= 10:
            return 4
        elif price <= 50:
            return 3
        elif price <= 100:
            return 2
        else:
            return 1

    def validate_symbol(self, symbol):
        """Validate if symbol exists and is tradeable"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol and symbol_info['status'] == 'TRADING':
                    return True
            return False
        except Exception as e:
            print(f"Error validating symbol {symbol}: {e}")
            return False

    def place_manual_long_trade(self, symbol, trade_amount):
        """Place a long trade with specified amount and automatic SL and TP"""
        try:
            print(f"\n{'='*50}")
            print(f"PLACING LONG TRADE")
            print(f"Symbol: {symbol}")
            print(f"Trade Amount: {trade_amount} USDT")
            print(f"{'='*50}")
            
            # Validate symbol
            if not self.validate_symbol(symbol):
                raise Exception(f"Symbol {symbol} is not valid or not tradeable")
            
            # Check balance
            balance = self.get_futures_balance()
            if balance < trade_amount:
                raise Exception(f"Insufficient balance. Available: {balance} USDT, Required: {trade_amount} USDT")
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                raise Exception(f"Could not get current price for {symbol}")
            
            print(f"Current price: {current_price}")
            
            # Calculate quantity
            quantity = self.calculate_quantity_from_amount(symbol, trade_amount, current_price)
            if not quantity:
                raise Exception(f"Could not calculate quantity for {symbol}")
            
            # Set leverage
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
                print(f"Leverage set to {self.leverage}x for {symbol}")
            except Exception as e:
                print(f"Warning: Could not set leverage: {e}")
            
            # Place market buy order
            market_order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            print(f"✅ Market long order placed successfully for {symbol}")
            print(f"Order ID: {market_order['orderId']}")
            
            # Get updated price after order execution
            entry_price = self.get_current_price(symbol)
            price_precision = self.get_price_precision(entry_price)
            
            # Calculate stop loss price
            stop_loss_price = entry_price * (1 - self.stop_loss_percent / 100)
            stop_loss_price = round(stop_loss_price, price_precision)
            
            # Calculate take profit price
            take_profit_price = entry_price * (1 + self.target_profit_percent / 100)
            take_profit_price = round(take_profit_price, price_precision)
            
            print(f"\n📊 TRADE SUMMARY:")
            print(f"Entry Price: {entry_price}")
            print(f"Quantity: {quantity}")
            print(f"Stop Loss: {stop_loss_price} (-{self.stop_loss_percent}%)")
            print(f"Take Profit: {take_profit_price} (+{self.target_profit_percent}%)")
            
            # Place stop loss order
            stop_loss_order = None
            try:
                stop_loss_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type='STOP_MARKET',
                    stopPrice=stop_loss_price,
                    closePosition='true'
                )
                print(f"✅ Stop loss order placed at {stop_loss_price}")
            except Exception as e:
                print(f"⚠️  Warning: Could not place stop loss: {e}")
            
            # Place take profit order
            take_profit_order = None
            try:
                take_profit_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type='LIMIT',
                    price=take_profit_price,
                    quantity=quantity,
                    timeInForce='GTC'
                )
                print(f"✅ Take profit order placed at {take_profit_price}")
            except Exception as e:
                print(f"⚠️  Warning: Could not place take profit: {e}")
            
            # Calculate actual trade value
            actual_trade_value = entry_price * quantity
            
            print(f"\n💰 FINANCIAL SUMMARY:")
            print(f"Actual Trade Value: {round(actual_trade_value, 2)} USDT")
            print(f"Potential Profit: {round(actual_trade_value * self.target_profit_percent / 100, 2)} USDT")
            print(f"Potential Loss: {round(actual_trade_value * self.stop_loss_percent / 100, 2)} USDT")
            print(f"Remaining Balance: {round(balance - trade_amount, 2)} USDT")
            
            return {
                'success': True,
                'market_order': market_order,
                'stop_loss_order': stop_loss_order,
                'take_profit_order': take_profit_order,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'quantity': quantity,
                'trade_amount': trade_amount
            }
            
        except Exception as e:
            error_msg = f"❌ Error placing long trade for {symbol}: {e}"
            print(error_msg)
            
            return {
                'success': False,
                'error': str(e)
            }

def get_user_input():
    """Get trading parameters from user input"""
    print("\n" + "="*60)
    print("MANUAL LONG POSITION TRADING")
    print("="*60)
    
    # Get symbol
    symbol = input("Enter coin symbol (e.g., BTCUSDT, ETHUSDT): ").upper().strip()
    if not symbol.endswith('USDT'):
        symbol += 'USDT'
    
    # Get trade amount
    while True:
        try:
            trade_amount = float(input("Enter trade amount in USDT: "))
            if trade_amount <= 0:
                print("Trade amount must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    return symbol, trade_amount

def main():
    """Main function to execute manual long trading"""
    try:
        # Initialize trader
        trader = ManualFuturesTrader()
        
        # Show current balance
        balance = trader.get_futures_balance()
        print(f"\n💰 Current Futures Balance: {balance} USDT")
        
        # Use predefined variables
        symbol = COIN_NAME.upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        trade_amount = TRADE_AMOUNT
        
        print(f"\n📋 CONFIGURED TRADE:")
        print(f"Symbol: {symbol}")
        print(f"Trade Amount: {trade_amount} USDT")
        print(f"Leverage: {trader.leverage}x")
        print(f"Stop Loss: {trader.stop_loss_percent}%")
        print(f"Take Profit: {trader.target_profit_percent}%")
        
        # Optional confirmation (comment out the next 4 lines if you want auto-execution)
        confirm = input("\nDo you want to proceed with this trade? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Trade cancelled.")
            return
        
        # Execute trade
        result = trader.place_manual_long_trade(symbol, trade_amount)
        
        if result['success']:
            print(f"\n🎉 Trade executed successfully!")
            print(f"Check your Binance Futures account for position details.")
        else:
            print(f"\n❌ Trade failed: {result['error']}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Trading cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
