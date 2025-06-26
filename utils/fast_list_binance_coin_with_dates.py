import requests
import json
from datetime import datetime
from binance.client import Client
import os
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()
load_dotenv('../.env')
load_dotenv('../bot/.env')

# Thread-local storage for clients
thread_local = threading.local()

def get_client():
    """Get a thread-local Binance client"""
    if not hasattr(thread_local, 'client'):
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        thread_local.client = Client(api_key, api_secret, tld='com')
    return thread_local.client

def get_futures_pairs_with_dates():
    """Get all active futures trading pairs with their listing dates"""
    print("🔍 Fetching futures pairs...")
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    pairs = []
    for symbol in data['symbols']:
        if (symbol['status'] == 'TRADING' and 
            symbol['contractType'] == 'PERPETUAL'):
            pairs.append({
                'symbol': symbol['symbol'],
                'baseAsset': symbol['baseAsset'],
                'quoteAsset': symbol['quoteAsset'],
                'contractType': symbol['contractType'],
                'listing_date': None
            })
    
    print(f"📊 Found {len(pairs)} active futures pairs")
    return pairs

def get_listing_date_for_symbol(symbol_data):
    """Get the listing date for a single symbol"""
    symbol = symbol_data['symbol']
    try:
        client = get_client()
        earliest_klines = client.futures_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str="1 Jan, 2017",
            limit=1
        )
        
        if earliest_klines and len(earliest_klines) > 0:
            timestamp = earliest_klines[0][0]
            listing_date = datetime.fromtimestamp(timestamp / 1000)
            symbol_data['listing_date'] = listing_date.strftime('%Y-%m-%d %H:%M:%S')
            return symbol_data, True
        
        return symbol_data, False
        
    except Exception as e:
        print(f"⚠️ Error for {symbol}: {e}")
        return symbol_data, False

def fetch_listing_dates_parallel(pairs, max_workers=10):
    """Fetch listing dates for all pairs using parallel processing"""
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("⚠️ No API credentials found. Listing dates will not be available.")
        return pairs
    
    print(f"🚀 Using {max_workers} parallel workers for faster processing")
    print(f"📅 Fetching listing dates for {len(pairs)} futures pairs...")
    print(f"🔄 Processing {len(pairs)} coins in total...")
    
    processed_count = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(get_listing_date_for_symbol, pair.copy()): pair for pair in pairs}
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            processed_count += 1
            try:
                result_pair, success = future.result()
                if success:
                    success_count += 1
                    # Update the original pair
                    original_pair = future_to_symbol[future]
                    original_pair['listing_date'] = result_pair['listing_date']
                    print(f"[{processed_count:3d}/{len(pairs)}] ✅ {result_pair['symbol']}: {result_pair['listing_date']}")
                else:
                    print(f"[{processed_count:3d}/{len(pairs)}] ❌ {result_pair['symbol']}: No data")
                
                # Show progress every 25 coins
                if processed_count % 25 == 0:
                    percentage = (processed_count / len(pairs)) * 100
                    print(f"🚀 Progress: {processed_count}/{len(pairs)} coins processed ({percentage:.1f}% complete)")
                
                # Show milestone progress
                if processed_count % 100 == 0:
                    remaining = len(pairs) - processed_count
                    print(f"🎯 Milestone: {processed_count} coins processed! ({remaining} coins remaining)")
                    
            except Exception as e:
                print(f"[{processed_count:3d}/{len(pairs)}] ❌ Error: {e}")
    
    print(f"✅ Completed! {success_count}/{len(pairs)} coins got listing dates")
    return pairs

def sort_pairs_by_date(pairs):
    """Sort pairs by listing date"""
    pairs_with_dates = [p for p in pairs if p['listing_date'] is not None]
    pairs_without_dates = [p for p in pairs if p['listing_date'] is None]
    
    pairs_with_dates.sort(key=lambda x: x['listing_date'])
    
    print(f"📊 Sorted {len(pairs_with_dates)} pairs by listing date")
    print(f"⚠️ {len(pairs_without_dates)} pairs without listing dates")
    
    return pairs_with_dates + pairs_without_dates

def save_to_json(pairs, filename):
    """Save pairs to JSON file"""
    with open(filename, 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"💾 Saved to {filename}")

def save_to_csv(pairs, filename):
    """Save pairs to CSV file for easy viewing"""
    import csv
    
    if not pairs:
        return
    
    fieldnames = pairs[0].keys()
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    
    print(f"💾 Saved to {filename}")

def print_summary(pairs, pair_type):
    """Print summary statistics"""
    print(f"\n📊 {pair_type.upper()} PAIRS SUMMARY")
    print("=" * 50)
    print(f"Total pairs: {len(pairs)}")
    
    pairs_with_dates = [p for p in pairs if p['listing_date'] is not None]
    if pairs_with_dates:
        print(f"Pairs with dates: {len(pairs_with_dates)}")
        print(f"Oldest listing: {pairs_with_dates[0]['symbol']} - {pairs_with_dates[0]['listing_date']}")
        print(f"Newest listing: {pairs_with_dates[-1]['symbol']} - {pairs_with_dates[-1]['listing_date']}")
        
        print(f"\n🕐 OLDEST 10 FUTURES PAIRS:")
        for i, pair in enumerate(pairs_with_dates[:10], 1):
            print(f"{i:2d}. {pair['symbol']:15s} - {pair['listing_date']}")
        
        print(f"\n🆕 NEWEST 10 FUTURES PAIRS:")
        for i, pair in enumerate(pairs_with_dates[-10:], len(pairs_with_dates)-9):
            print(f"{i:2d}. {pair['symbol']:15s} - {pair['listing_date']}")

def main():
    """Main function"""
    start_time = time.time()
    
    print("=" * 80)
    print("⚡ FAST BINANCE COIN LISTING ANALYZER WITH DATES")
    print("🚀 Using parallel processing for maximum speed")
    print("=" * 80)
    
    # Get futures pairs
    print("\n🚀 PROCESSING FUTURES PAIRS")
    print("-" * 40)
    futures_pairs = get_futures_pairs_with_dates()
    print(f"🎯 TOTAL COINS TO PROCESS: {len(futures_pairs)}")
    print(f"⏱️  Estimated time: ~{len(futures_pairs) / 50:.1f} minutes (with parallel processing)")
    print("-" * 40)
    
    futures_pairs = fetch_listing_dates_parallel(futures_pairs, max_workers=10)
    futures_pairs = sort_pairs_by_date(futures_pairs)
    
    # Save results
    print("\n💾 SAVING RESULTS")
    print("-" * 40)
    
    save_to_json(futures_pairs, 'futures_with_dates_fast.json')
    save_to_csv(futures_pairs, 'futures_with_dates_fast.csv')
    
    print_summary(futures_pairs, "futures")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 Analysis completed in {total_time:.1f} seconds!")
    print(f"📁 Files saved:")
    print(f"   - futures_with_dates_fast.json & .csv")
    print(f"⚡ Speed: {len(futures_pairs) / total_time:.1f} coins per second")

if __name__ == "__main__":
    main() 