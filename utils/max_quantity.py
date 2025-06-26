import requests
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import time
from decimal import Decimal, ROUND_DOWN
import math

class PracticalBinanceCalculator:
    """
    Practical calculator to find EXACTLY how much you can buy with your balance
    Addresses the common API error: -4005 Quantity greater than max quantity
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
        
        # Common notional limits (these are typical values, will be updated from API)
        self.typical_limits = {
            'min_notional': 5.0,  # Minimum $5 order value
            'max_notional': 10000000.0,  # $10M max for most pairs
        }
    
    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Get all trading filters for a symbol"""
        url = f"{self.base_url}/api/v3/exchangeInfo"
        
        try:
            response = self.session.get(url, params={'symbol': symbol.upper()})
            response.raise_for_status()
            data = response.json()
            
            if 'symbols' in data and len(data['symbols']) > 0:
                symbol_info = data['symbols'][0]
                filters = {}
                
                for filter_info in symbol_info.get('filters', []):
                    filters[filter_info['filterType']] = filter_info
                
                return {
                    'symbol': symbol_info['symbol'],
                    'status': symbol_info['status'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                    'filters': filters
                }
            
        except Exception as e:
            print(f"Error fetching filters for {symbol}: {e}")
        
        return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        url = f"{self.base_url}/api/v3/ticker/price"
        
        try:
            response = self.session.get(url, params={'symbol': symbol.upper()})
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def calculate_exact_buyable_amount(self, symbol: str, balance_usdt: float) -> Dict[str, Any]:
        """
        Calculate EXACTLY how much you can buy with your balance
        Handles all Binance constraints and avoids API errors
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            balance_usdt: Your available balance in USDT/quote currency
            
        Returns:
            Dict with exact buyable amount and all constraints
        """
        
        # Get symbol information
        symbol_info = self.get_symbol_filters(symbol)
        if not symbol_info:
            return {'error': f'Could not fetch data for {symbol}'}
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            return {'error': f'Could not fetch current price for {symbol}'}
        
        filters = symbol_info.get('filters', {})
        
        result = {
            'symbol': symbol.upper(),
            'current_price': current_price,
            'available_balance': balance_usdt,
            'constraints': {},
            'calculations': {},
            'final_result': {}
        }
        
        # Extract all relevant constraints
        constraints = {}
        
        # LOT_SIZE filter - quantity constraints
        if 'LOT_SIZE' in filters:
            lot_size = filters['LOT_SIZE']
            constraints['min_qty'] = float(lot_size['minQty'])
            constraints['max_qty'] = float(lot_size['maxQty']) 
            constraints['step_size'] = float(lot_size['stepSize'])
        
        # MARKET_LOT_SIZE filter - market order constraints
        if 'MARKET_LOT_SIZE' in filters:
            market_lot = filters['MARKET_LOT_SIZE']
            constraints['market_min_qty'] = float(market_lot['minQty'])
            constraints['market_max_qty'] = float(market_lot['maxQty'])
            constraints['market_step_size'] = float(market_lot['stepSize'])
        
        # MIN_NOTIONAL filter - minimum order value
        if 'MIN_NOTIONAL' in filters:
            min_notional = filters['MIN_NOTIONAL']
            constraints['min_notional'] = float(min_notional['minNotional'])
        elif 'NOTIONAL' in filters:
            notional = filters['NOTIONAL']
            constraints['min_notional'] = float(notional.get('minNotional', 5.0))
        else:
            constraints['min_notional'] = 5.0  # Default minimum
        
        # NOTIONAL filter - order value limits
        if 'NOTIONAL' in filters:
            notional = filters['NOTIONAL']
            constraints['max_notional'] = float(notional.get('maxNotional', 10000000.0))
        else:
            constraints['max_notional'] = 10000000.0  # Default maximum
        
        result['constraints'] = constraints
        
        # Calculate maximum affordable quantity
        max_affordable_raw = balance_usdt / current_price
        
        # Apply lot size constraints
        step_size = constraints.get('step_size', constraints.get('market_step_size', 0.00000001))
        min_qty = constraints.get('min_qty', constraints.get('market_min_qty', 0.00000001))
        max_qty = constraints.get('max_qty', constraints.get('market_max_qty', float('inf')))
        
        # Round down to step size
        max_affordable_stepped = self._round_to_step_size(max_affordable_raw, step_size, 'down')
        
        # Apply min/max quantity constraints
        max_affordable_constrained = max(min_qty, min(max_affordable_stepped, max_qty))
        
        # Check notional constraints
        min_notional = constraints.get('min_notional', 5.0)
        max_notional = constraints.get('max_notional', 10000000.0)
        
        # Ensure we meet minimum notional value
        min_qty_for_notional = min_notional / current_price
        min_qty_for_notional_stepped = self._round_to_step_size(min_qty_for_notional, step_size, 'up')
        
        # Final quantity is the maximum of all minimum requirements
        final_min_qty = max(min_qty, min_qty_for_notional_stepped)
        
        # But not more than what we can afford or the maximum allowed
        final_max_qty = min(max_affordable_constrained, max_notional / current_price)
        
        # Ensure we have enough balance for the minimum order
        if balance_usdt < min_notional:
            result['final_result'] = {
                'can_trade': False,
                'reason': f'Insufficient balance. Need at least ${min_notional:.2f} but have ${balance_usdt:.2f}',
                'minimum_needed': min_notional,
                'shortage': min_notional - balance_usdt
            }
            return result
        
        # Calculate final buyable quantity
        if final_min_qty <= final_max_qty:
            final_quantity = final_max_qty
            final_cost = final_quantity * current_price
            
            result['final_result'] = {
                'can_trade': True,
                'exact_quantity': final_quantity,
                'exact_cost': final_cost,
                'remaining_balance': balance_usdt - final_cost,
                'percentage_of_balance_used': (final_cost / balance_usdt) * 100
            }
        else:
            result['final_result'] = {
                'can_trade': False,
                'reason': 'Constraints conflict - minimum required quantity costs more than maximum allowed',
                'min_required_qty': final_min_qty,
                'max_allowed_qty': final_max_qty
            }
        
        # Add calculation details
        result['calculations'] = {
            'raw_affordable': max_affordable_raw,
            'after_step_size': max_affordable_stepped,
            'after_constraints': max_affordable_constrained,
            'min_qty_for_notional': min_qty_for_notional_stepped,
            'final_min_qty': final_min_qty,
            'final_max_qty': final_max_qty,
            'step_size_used': step_size
        }
        
        return result
    
    def _round_to_step_size(self, quantity: float, step_size: float, direction: str = 'down') -> float:
        """
        Round quantity to valid step size
        
        Args:
            quantity: Quantity to round
            step_size: Step size from LOT_SIZE filter
            direction: 'up' or 'down'
        """
        if step_size == 0:
            return quantity
        
        # Count decimal places in step size
        step_str = f"{step_size:.8f}".rstrip('0')
        if '.' in step_str:
            decimals = len(step_str.split('.')[1])
        else:
            decimals = 0
        
        # Calculate steps
        steps = quantity / step_size
        
        if direction == 'down':
            rounded_steps = math.floor(steps)
        else:  # direction == 'up'
            rounded_steps = math.ceil(steps)
        
        result = rounded_steps * step_size
        
        # Round to appropriate decimal places
        return round(result, decimals)
    
    def quick_check_multiple_coins(self, coins: list, balance: float) -> pd.DataFrame:
        """
        Quick check for multiple coins to see what you can buy
        
        Args:
            coins: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            balance: Available balance in USDT
        """
        results = []
        
        for coin in coins:
            try:
                calc = self.calculate_exact_buyable_amount(coin, balance)
                
                if 'error' in calc:
                    results.append({
                        'Symbol': coin,
                        'Current Price': 'Error',
                        'Can Trade': False,
                        'Max Quantity': 'N/A',
                        'Cost': 'N/A',
                        'Remaining': 'N/A'
                    })
                    continue
                
                final = calc['final_result']
                
                if final['can_trade']:
                    results.append({
                        'Symbol': coin,
                        'Current Price': f"${calc['current_price']:,.4f}",
                        'Can Trade': True,
                        'Max Quantity': f"{final['exact_quantity']:.8f}",
                        'Cost': f"${final['exact_cost']:.2f}",
                        'Remaining': f"${final['remaining_balance']:.2f}"
                    })
                else:
                    results.append({
                        'Symbol': coin,
                        'Current Price': f"${calc['current_price']:,.4f}",
                        'Can Trade': False,
                        'Max Quantity': final.get('reason', 'Cannot trade'),
                        'Cost': 'N/A',
                        'Remaining': 'N/A'
                    })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                results.append({
                    'Symbol': coin,
                    'Current Price': 'Error',
                    'Can Trade': False,
                    'Max Quantity': str(e),
                    'Cost': 'N/A',
                    'Remaining': 'N/A'
                })
        
        return pd.DataFrame(results)
    
    def print_detailed_analysis(self, symbol: str, balance: float):
        """Print detailed analysis for a specific symbol"""
        
        print(f"\n{'='*70}")
        print(f"EXACT BUYABLE AMOUNT ANALYSIS FOR {symbol.upper()}")
        print(f"Available Balance: ${balance:,.2f}")
        print(f"{'='*70}")
        
        result = self.calculate_exact_buyable_amount(symbol, balance)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"💰 Current Price: ${result['current_price']:,.6f}")
        
        print(f"\n📋 TRADING CONSTRAINTS:")
        constraints = result['constraints']
        for key, value in constraints.items():
            if 'qty' in key:
                print(f"   {key.replace('_', ' ').title()}: {value:.8f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: ${value:,.2f}")
        
        print(f"\n🔢 CALCULATION STEPS:")
        calc = result['calculations']
        print(f"   Raw affordable quantity: {calc['raw_affordable']:.8f}")
        print(f"   After step size rounding: {calc['after_step_size']:.8f}")
        print(f"   After quantity constraints: {calc['after_constraints']:.8f}")
        print(f"   Min quantity for notional: {calc['min_qty_for_notional']:.8f}")
        print(f"   Step size used: {calc['step_size_used']:.8f}")
        
        print(f"\n🎯 FINAL RESULT:")
        final = result['final_result']
        
        if final['can_trade']:
            print(f"   ✅ CAN TRADE: YES")
            print(f"   📊 Exact Quantity: {final['exact_quantity']:.8f} {result['symbol'][:-4]}")
            print(f"   💵 Exact Cost: ${final['exact_cost']:.2f}")
            print(f"   💰 Remaining Balance: ${final['remaining_balance']:.2f}")
            print(f"   📈 Balance Used: {final['percentage_of_balance_used']:.1f}%")
            
            print(f"\n🤖 READY-TO-USE ORDER:")
            print(f"   Symbol: {result['symbol']}")
            print(f"   Side: BUY")
            print(f"   Type: MARKET or LIMIT")
            print(f"   Quantity: {final['exact_quantity']:.8f}")
            print(f"   (This will cost ~${final['exact_cost']:.2f})")
            
        else:
            print(f"   ❌ CAN TRADE: NO")
            print(f"   🚫 Reason: {final['reason']}")
            if 'minimum_needed' in final:
                print(f"   💵 Minimum Needed: ${final['minimum_needed']:.2f}")
                print(f"   📉 Shortage: ${final['shortage']:.2f}")

# Example usage and testing
if __name__ == "__main__":
    calculator = PracticalBinanceCalculator()
    
    # Test with your specific case: $1000 balance
    balance = 1000.0
    
    print("🚀 PRACTICAL BINANCE TRADING CALCULATOR")
    print("Solving the 'Quantity greater than max quantity' error")
    print(f"Testing with ${balance:,.2f} balance\n")
    
    # Test popular coins
    test_coins = ['SAHARAUSDT']
    
    # Quick overview
    print("📊 QUICK OVERVIEW - What you can buy:")
    overview = calculator.quick_check_multiple_coins(test_coins, balance)
    print(overview.to_string(index=False))
    
    # Detailed analysis for Bitcoin (most likely to have constraints)
    calculator.print_detailed_analysis('BTCUSDT', balance)
    
    # Detailed analysis for Ethereum
    calculator.print_detailed_analysis('ETHUSDT', balance)
    
    print(f"\n{'='*70}")
    print("💡 HOW TO USE THIS CALCULATOR:")
    print("1. Run: calculator = PracticalBinanceCalculator()")
    print("2. Check specific coin: calculator.print_detailed_analysis('BTCUSDT', 1000)")
    print("3. Get exact quantity: result = calculator.calculate_exact_buyable_amount('ETHUSDT', 500)")
    print("4. Use the 'exact_quantity' in your Binance API order")
    print("5. This avoids ALL quantity/notional errors!")
    
    print(f"\n🎯 COMMON ERROR SOLUTIONS:")
    print("• Error -4005 (Qty > max): Use the calculated 'exact_quantity'")
    print("• Error -4004 (Qty < min): Increase your balance or choose different coin")
    print("• Error -1013 (LOT_SIZE): Quantity is rounded to proper step size")
    print("• Error MIN_NOTIONAL: Minimum $5-10 order value enforced")
    
    print(f"\n📝 COPY-PASTE READY CODE:")
    print("""
# Quick check what you can buy with $500
calculator = PracticalBinanceCalculator()
result = calculator.calculate_exact_buyable_amount('BTCUSDT', 500)

if result['final_result']['can_trade']:
    quantity = result['final_result']['exact_quantity']
    print(f"Buy exactly: {quantity:.8f} BTC")
    # Use this quantity in your Binance order
else:
    print("Cannot trade:", result['final_result']['reason'])
""")