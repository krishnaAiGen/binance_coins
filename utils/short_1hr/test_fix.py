#!/usr/bin/env python3

from backtest_ema_short import EMAShortBacktester

def test_single_symbol():
    print("🧪 Testing EMA Short Signal Detection Fix")
    
    backtester = EMAShortBacktester()
    print(f"Found {len(backtester.symbols)} symbols")
    
    if backtester.symbols:
        # Test first symbol
        symbol = backtester.symbols[0]
        print(f"\n🎯 Testing symbol: {symbol}")
        
        # Load data
        df = backtester.load_symbol_data(symbol)
        if df is not None:
            print(f"✅ Loaded {len(df)} rows of data")
            
            # Show first few hours of price vs EMA data
            print("\n📊 First 4 hours data:")
            cols = ['hours_from_launch', 'close', 'ema_5']
            if all(col in df.columns for col in cols):
                for i in range(min(4, len(df))):
                    row = df.iloc[i]
                    close_val = row['close']
                    ema5_val = row['ema_5']
                    signal = "SHORT" if close_val < ema5_val else "NO"
                    print(f"   Hour {i}: Close={close_val:.6f}, EMA5={ema5_val:.6f}, Signal={signal}")
            
            # Test signal detection at hour 2
            print(f"\n🔍 Testing signal at hour 2...")
            trade = backtester.simulate_ema_short_trade(df, symbol, ema_period=5, target_pct=15.0, stop_loss_pct=5.0, start_hour=2)
            
            if trade:
                print(f"📈 Trade Result:")
                print(f"   Entry Signal: {trade.get('entry_signal', False)}")
                print(f"   Signal Reason: {trade.get('signal_reason', 'N/A')}")
                print(f"   Exit Reason: {trade.get('exit_reason', 'N/A')}")
                print(f"   Profit USD: ${trade.get('profit_usd', 0):.2f}")
                print(f"   Profit %: {trade.get('profit_pct', 0):.2f}%")
                
                if trade.get('entry_signal', False):
                    print(f"   ✅ Signal detected successfully!")
                else:
                    print(f"   ❌ No signal detected")
            else:
                print("❌ No trade result returned")
        else:
            print("❌ Failed to load data")
    else:
        print("❌ No symbols found")

if __name__ == "__main__":
    test_single_symbol()
