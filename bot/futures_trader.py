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
    
    def get_available_futures_balance(self):
        """Get available USDT balance for trading (excluding locked in positions)"""
        try:
            balance_info = self.client.futures_account_balance()
            for asset in balance_info:
                if asset['asset'] == 'USDT':
                    # Use availableBalance instead of balance
                    available_balance = float(asset.get('availableBalance', asset['balance']))
                    total_balance = float(asset['balance'])
                    locked_balance = total_balance - available_balance
                    
                    print(f"💰 Futures Balance Summary:")
                    print(f"   Total USDT: ${total_balance:.2f}")
                    print(f"   Available: ${available_balance:.2f}")
                    print(f"   Locked: ${locked_balance:.2f}")
                    
                    return available_balance
            return 0.0
        except Exception as e:
            print(f"Error getting available futures balance: {e}")
            self.slack.post_error_to_slack(f"Error getting available futures balance: {e}")
            return 0.0

    def get_current_price(self, symbol):
        """Get current price for a futures symbol"""
        try:
            # First try futures mark price (most reliable for futures)
            mark_price = self.client.futures_mark_price(symbol=symbol)
            return float(mark_price['markPrice'])
        except Exception as e1:
            try:
                # Fallback to futures ticker price
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except Exception as e2:
                try:
                    # Last fallback: spot price (for newly listed symbols)
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    return float(ticker['price'])
                except Exception as e3:
                    print(f"Error getting current price for {symbol}:")
                    print(f"  Futures mark price: {e1}")
                    print(f"  Futures ticker: {e2}")
                    print(f"  Spot ticker: {e3}")
                    return None

    def is_valid_futures_symbol(self, symbol):
        """Check if symbol exists on Binance Futures"""
        try:
            exchange_info = self.client.futures_exchange_info()
            valid_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            return symbol in valid_symbols
        except Exception as e:
            print(f"Error checking symbol validity for {symbol}: {e}")
            return False

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
            
            # Force integer quantity always
            rounded_quantity = int(round(quantity))
            
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
            
            # Try to place single order first
            try:
                return self._place_single_long_trade(symbol, quantity, current_price)
            except Exception as e:
                error_str = str(e)
                print(f"Single order failed: {error_str}")
                
                # Check if it's a quantity-related error or any other error that might benefit from splitting
                if "quantity" in error_str.lower() or "size" in error_str.lower() or "notional" in error_str.lower():
                    print("Attempting to break order into smaller chunks...")
                    return self._place_multiple_long_trades(symbol, quantity, current_price)
                else:
                    # Re-raise the exception if it's not quantity-related
                    raise e
            
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

    def place_fixed_amount_trade(self, symbol, usd_amount):
        """Place a trade for a specific USD amount without order splitting"""
        try:
            print(f"Attempting to place ${usd_amount} trade for {symbol}")
            
            # First validate that symbol exists on futures
            if not self.is_valid_futures_symbol(symbol):
                raise Exception(f"Symbol {symbol} is not available on Binance Futures or not trading")
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                raise Exception(f"Could not get current price for {symbol}")
            
            # Calculate quantity based on USD amount and leverage
            quantity = (usd_amount * self.leverage) / current_price
            quantity = int(round(quantity))  # Force integer quantity
            
            if quantity <= 0:
                raise Exception(f"Calculated quantity is zero or negative: {quantity}")
            
            print(f"Calculated quantity for ${usd_amount}: {quantity}")
            
            # Set leverage
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
                print(f"Leverage set to {self.leverage}x for {symbol}")
            except Exception as e:
                print(f"Warning: Could not set leverage: {e}")
            
            # Place the trade (without order splitting)
            return self._place_single_long_trade(symbol, quantity, current_price)
            
        except Exception as e:
            error_msg = f"Error placing fixed amount trade for {symbol}: {e}"
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

    def _place_single_long_trade(self, symbol, quantity, current_price):
        """Place a single long trade"""
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
        
        # Calculate stop loss and take profit prices
        stop_loss_price = entry_price * (1 - self.stop_loss_percent / 100)
        stop_loss_price = round(stop_loss_price, price_precision)
        
        take_profit_price = entry_price * (1 + self.target_profit_percent / 100)
        take_profit_price = round(take_profit_price, price_precision)
        
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
            print(f"Stop loss order placed at {stop_loss_price}")
        except Exception as e:
            print(f"Warning: Could not place stop loss: {e}")
        
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
            print(f"Take profit order placed at {take_profit_price}")
        except Exception as e:
            print(f"Warning: Could not place take profit: {e}")
        
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

    def _place_multiple_long_trades(self, symbol, total_quantity, current_price):
        """Break large order into smaller chunks and place multiple trades"""
        MAX_ORDER_VALUE = 200.0  # Maximum value per order in USDT
        
        # Calculate total trade value
        total_trade_value = total_quantity * current_price
        print(f"Total trade value: ${total_trade_value:.2f}")
        
        # Calculate number of orders needed
        num_orders = math.ceil(total_trade_value / MAX_ORDER_VALUE)
        print(f"Breaking into {num_orders} orders (max ${MAX_ORDER_VALUE} each)")
        
        # Calculate quantity per order
        base_quantity_per_order = total_quantity / num_orders
        precision = 0  # Force integer quantities always
        
        successful_orders = []
        failed_orders = []
        
        for i in range(num_orders):
            try:
                # Calculate quantity for this order (handle remainder for last order)
                if i == num_orders - 1:
                    # Last order gets any remaining quantity
                    remaining_quantity = total_quantity - sum([order['quantity'] for order in successful_orders])
                    order_quantity = int(round(remaining_quantity, precision))
                else:
                    order_quantity = int(round(base_quantity_per_order, precision))
                
                if order_quantity <= 0:
                    continue
                
                print(f"\n--- Placing order {i+1}/{num_orders} ---")
                print(f"Order quantity: {order_quantity}")
                
                # Place market buy order
                market_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=order_quantity
                )
                
                print(f"✅ Market order {i+1} placed successfully")
                
                # Get entry price for this order
                entry_price = self.get_current_price(symbol)
                price_precision = self.get_price_precision(entry_price)
                
                # Calculate stop loss and take profit prices
                stop_loss_price = entry_price * (1 - self.stop_loss_percent / 100)
                stop_loss_price = round(stop_loss_price, price_precision)
                
                take_profit_price = entry_price * (1 + self.target_profit_percent / 100)
                take_profit_price = round(take_profit_price, price_precision)
                
                # Place stop loss order for this position
                stop_loss_order = None
                try:
                    stop_loss_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type='STOP_MARKET',
                        stopPrice=stop_loss_price,
                        quantity=order_quantity
                    )
                    print(f"✅ Stop loss order {i+1} placed at {stop_loss_price}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not place stop loss for order {i+1}: {e}")
                
                # Place take profit order for this position
                take_profit_order = None
                try:
                    take_profit_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type='LIMIT',
                        price=take_profit_price,
                        quantity=order_quantity,
                        timeInForce='GTC'
                    )
                    print(f"✅ Take profit order {i+1} placed at {take_profit_price}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not place take profit for order {i+1}: {e}")
                
                # Calculate trade amount for this order
                order_trade_amount = entry_price * order_quantity
                
                # Store successful order info
                order_info = {
                    'order_number': i + 1,
                    'market_order': market_order,
                    'stop_loss_order': stop_loss_order,
                    'take_profit_order': take_profit_order,
                    'entry_price': entry_price,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'quantity': order_quantity,
                    'trade_amount': round(order_trade_amount, 2)
                }
                
                successful_orders.append(order_info)
                
                # Prepare individual trade info for notification
                trade_info = {
                    'success': True,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'quantity': order_quantity,
                    'stop_loss_price': stop_loss_price,
                    'stop_loss_pct': self.stop_loss_percent,
                    'take_profit_price': take_profit_price,
                    'profit_target_pct': self.target_profit_percent,
                    'leverage': self.leverage,
                    'trade_amount': round(order_trade_amount, 2),
                    'order_number': i + 1,
                    'total_orders': num_orders
                }
                
                # Send notification for each individual trade
                self.slack.post_trade_notification(trade_info)
                
                print(f"📊 Order {i+1} Summary:")
                print(f"   Entry Price: {entry_price}")
                print(f"   Quantity: {order_quantity}")
                print(f"   Trade Amount: ${order_trade_amount:.2f}")
                print(f"   Stop Loss: {stop_loss_price} (-{self.stop_loss_percent}%)")
                print(f"   Take Profit: {take_profit_price} (+{self.target_profit_percent}%)")
                
            except Exception as e:
                error_msg = f"Failed to place order {i+1}: {e}"
                print(f"❌ {error_msg}")
                failed_orders.append({
                    'order_number': i + 1,
                    'error': str(e)
                })
        
        # Summary
        print(f"\n🎯 MULTI-ORDER SUMMARY:")
        print(f"Total orders attempted: {num_orders}")
        print(f"Successful orders: {len(successful_orders)}")
        print(f"Failed orders: {len(failed_orders)}")
        
        if successful_orders:
            total_successful_quantity = sum([order['quantity'] for order in successful_orders])
            total_successful_amount = sum([order['trade_amount'] for order in successful_orders])
            avg_entry_price = sum([order['entry_price'] * order['quantity'] for order in successful_orders]) / total_successful_quantity
            
            print(f"Total successful quantity: {total_successful_quantity}")
            print(f"Total successful amount: ${total_successful_amount:.2f}")
            print(f"Average entry price: {avg_entry_price:.4f}")
            
            return {
                'success': True,
                'orders': successful_orders,
                'failed_orders': failed_orders,
                'total_quantity': total_successful_quantity,
                'total_amount': total_successful_amount,
                'average_entry_price': avg_entry_price,
                'num_successful_orders': len(successful_orders),
                'num_failed_orders': len(failed_orders)
            }
        else:
            return {
                'success': False,
                'error': f"All {num_orders} orders failed",
                'failed_orders': failed_orders
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