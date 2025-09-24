#!/usr/bin/env python3

import os
from futures_trader import FuturesTrader

def test_symbol_validation():
    """Test symbol validation and price fetching"""
    
    print("🧪 Testing Symbol Validation and Price Fetching...")
    
    # Initialize trader
    try:
        trader = FuturesTrader()
        print("✅ FuturesTrader initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing trader: {e}")
        return
    
    # Test symbols
    test_symbols = [
        'BTCUSDT',      # Should work (popular futures symbol)
        'ETHUSDT',      # Should work (popular futures symbol)
        'BLESSUSDT',    # Might not exist (the problematic one)
        'INVALIDUSDT',  # Definitely doesn't exist
    ]
    
    print(f"\n📊 Testing {len(test_symbols)} symbols...")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing: {symbol}")
        
        # Test symbol validation
        try:
            is_valid = trader.is_valid_futures_symbol(symbol)
            print(f"   Valid on futures: {'✅ YES' if is_valid else '❌ NO'}")
        except Exception as e:
            print(f"   Validation error: {e}")
            continue
        
        if not is_valid:
            print(f"   ⏭️  Skipping price check (invalid symbol)")
            continue
        
        # Test price fetching
        try:
            price = trader.get_current_price(symbol)
            if price:
                print(f"   Current price: ${price:.6f}")
            else:
                print(f"   ❌ Could not get price")
        except Exception as e:
            print(f"   Price error: {e}")
    
    print(f"\n✅ Symbol validation test complete!")

if __name__ == "__main__":
    test_symbol_validation()
