#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Futures Trading Module for New Listing Bot
Handles futures trading operations with automatic SL and TP
"""

import os
import json
import math
from datetime import datetime
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from dotenv import load_dotenv
from slack_notifier import SlackNotifier

# Load environment variables
load_dotenv()

class FuturesTrader:
    def __init__(self):
        """Initialize the Futures Trader with API credentials from environment variables"""
        try:
            # Get API credentials from environment variables
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in environment variables")
            
            # Initialize the Binance client
            self.client = Client(api_key, api_secret, tld='com')
            
            # Load trading parameters from environment
            self.balance_percentage = float(os.getenv('BALANCE_PERCENTAGE', 90))
            self.target_profit_percent = float(os.getenv('TARGET_PROFIT_PERCENT', 15))
            self.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', 2))
            self.leverage = int(os.getenv('LEVERAGE', 3))
            
            # Initialize Slack notifier
            self.slack = SlackNotifier()
            
            print("Futures trader initialized successfully")
            
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
            self.slack.post_error_to_slack(f"Error getting futures balance: {e}")
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

    def calculate_quantity(self, symbol, current_price):
        """Calculate quantity based on balance percentage and current price"""
        try:
            balance = self.get_futures_balance()
            if balance <= 0:
                print("Insufficient balance")
                return None
            
            # Use percentage of balance for trading
            trade_balance = balance * (self.balance_percentage / 100)
            print(f"Total balance: {balance} USDT")
            print(f"Using {self.balance_percentage}% of balance: {trade_balance} USDT")
            
            # Calculate quantity with leverage
            quantity = (trade_balance * self.leverage) / current_price
            
            # Get symbol precision and round quantity
            precision = self.get_symbol_precision(symbol)
            rounded_quantity = round(quantity, precision)
            
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

    def place_long_trade(self, symbol):
        """Place a long trade with automatic SL and TP"""
        try:
            print(f"Attempting to place long trade for {symbol}")
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                raise Exception(f"Could not get current price for {symbol}")
            
            # Calculate quantity
            quantity = self.calculate_quantity(symbol, current_price)
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
            
            print(f"Market long order placed successfully for {symbol}")
            print(f"Order details: {market_order}")
            
            # Get updated price after order execution
            entry_price = self.get_current_price(symbol)
            price_precision = self.get_price_precision(entry_price)
            
            # Calculate stop loss price
            stop_loss_price = entry_price * (1 - self.stop_loss_percent / 100)
            stop_loss_price = round(stop_loss_price, price_precision)
            
            # Calculate take profit price
            take_profit_price = entry_price * (1 + self.target_profit_percent / 100)
            take_profit_price = round(take_profit_price, price_precision)
            
            # Place stop loss order
            try:
                stop_loss_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type='STOP_MARKET',
                    stopPrice=stop_loss_price,
                    closePosition='true'
                )
                print(f"Stop loss order placed at {stop_loss_price}")
            except Exception as e:
                print(f"Warning: Could not place stop loss: {e}")
                stop_loss_order = None
            
            # Place take profit order
            try:
                take_profit_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type='LIMIT',
                    price=take_profit_price,
                    quantity=quantity,
                    timeInForce='GTC'
                )
                print(f"Take profit order placed at {take_profit_price}")
            except Exception as e:
                print(f"Warning: Could not place take profit: {e}")
                take_profit_order = None
            
            # Calculate trade amount
            trade_amount = entry_price * quantity
            
            # Prepare trade info for notification
            trade_info = {
                'success': True,
                'symbol': symbol,
                'entry_price': entry_price,
                'quantity': quantity,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': self.stop_loss_percent,
                'take_profit_price': take_profit_price,
                'profit_target_pct': self.target_profit_percent,
                'leverage': self.leverage,
                'trade_amount': round(trade_amount, 2)
            }
            
            # Send success notification to Slack
            self.slack.post_trade_notification(trade_info)
            
            return {
                'success': True,
                'market_order': market_order,
                'stop_loss_order': stop_loss_order,
                'take_profit_order': take_profit_order,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'quantity': quantity
            }
            
        except Exception as e:
            error_msg = f"Error placing long trade for {symbol}: {e}"
            print(error_msg)
            
            # Send error notification to Slack
            trade_info = {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
            self.slack.post_trade_notification(trade_info)
            
            return {
                'success': False,
                'error': str(e)
            }

    def check_order_status(self, symbol, order_id):
        """Check if an order is still open"""
        try:
            open_orders = self.client.futures_get_open_orders(symbol=symbol)
            for order in open_orders:
                if order['orderId'] == order_id:
                    return 'OPEN'
            return 'FILLED'
        except Exception as e:
            print(f"Error checking order status: {e}")
            return 'ERROR'

    def cancel_order(self, symbol, order_id):
        """Cancel an open order"""
        try:
            result = self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            print(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            return False 