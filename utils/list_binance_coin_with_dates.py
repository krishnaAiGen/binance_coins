import requests
import json
from datetime import datetime
from binance.client import Client
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
load_dotenv('../.env')
load_dotenv('../bot/.env')

def get_spot_pairs_with_dates():
    """Get all active spot trading pairs with their listing dates"""
    print("🔍 Fetching spot pairs...")
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    pairs = []
    for symbol in data['symbols']:
        if symbol['status'] == 'TRADING':
            pairs.append({
                'symbol': symbol['symbol'],
                'baseAsset': symbol['baseAsset'],
                'quoteAsset': symbol['quoteAsset'],
                'listing_date': None  # Will be filled later
            })
    
    print(f"📊 Found {len(pairs)} active spot pairs")
    return pairs

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
                'listing_date': None  # Will be filled later
            })
    
    print(f"📊 Found {len(pairs)} active futures pairs")
    return pairs

def get_listing_date(client, symbol, is_futures=False):
    """Get the listing date for a symbol by finding first available kline"""
    try:
        if is_futures:
            earliest_klines = client.futures_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                start_str="1 Jan, 2017",  # Start from early date
                limit=1
            )
        else:
            earliest_klines = client.get_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                start_str="1 Jan, 2017",  # Start from early date
                limit=1
            )
        
        if earliest_klines and len(earliest_klines) > 0:
            timestamp = earliest_klines[0][0]
            listing_date = datetime.fromtimestamp(timestamp / 1000)
            return listing_date.strftime('%Y-%m-%d %H:%M:%S')
        
        return None
        
    except Exception as e:
        print(f"⚠️ Could not get listing date for {symbol}: {e}")
        return None

def fetch_listing_dates(pairs, is_futures=False):
    """Fetch listing dates for all pairs"""
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("⚠️ No API credentials found. Listing dates will not be available.")
        print("To get listing dates, set BINANCE_API_KEY and BINANCE_API_SECRET in .env file")
        return pairs
    
    try:
        client = Client(api_key, api_secret, tld='com')
        print("✅ Binance client initialized")
    except Exception as e:
        print(f"❌ Could not initialize Binance client: {e}")
        return pairs
    
    pair_type = "futures" if is_futures else "spot"
    print(f"📅 Fetching listing dates for {len(pairs)} {pair_type} pairs...")
    print(f"🔄 Processing {len(pairs)} coins in total...")
    
    for i, pair in enumerate(pairs, 1):
        symbol = pair['symbol']
        print(f"[{i:3d}/{len(pairs)}] Processing coin {i} of {len(pairs)}: {symbol}...", end=" ")
        
        listing_date = get_listing_date(client, symbol, is_futures)
        pair['listing_date'] = listing_date
        
        if listing_date:
            print(f"✅ {listing_date}")
        else:
            print("❌ Not available")
        
        # Reduced delay to speed up processing
        time.sleep(0.02)  # 20ms instead of 100ms
        
        # Save progress every 50 pairs with coin count
        if i % 50 == 0:
            percentage = (i / len(pairs)) * 100
            print(f"🚀 Progress: {i}/{len(pairs)} coins processed ({percentage:.1f}% complete)")
        
        # Show milestone progress
        if i % 100 == 0:
            print(f"🎯 Milestone: {i} coins processed! ({len(pairs) - i} coins remaining)")
    
    return pairs

def sort_pairs_by_date(pairs):
    """Sort pairs by listing date"""
    # Separate pairs with and without dates
    pairs_with_dates = [p for p in pairs if p['listing_date'] is not None]
    pairs_without_dates = [p for p in pairs if p['listing_date'] is None]
    
    # Sort pairs with dates
    pairs_with_dates.sort(key=lambda x: x['listing_date'])
    
    print(f"📊 Sorted {len(pairs_with_dates)} pairs by listing date")
    print(f"⚠️ {len(pairs_without_dates)} pairs without listing dates")
    
    # Combine: dated pairs first, then undated pairs
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
    
    # Get all keys from the first pair
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
        
        # Show first 10 and last 10
        print(f"\n🕐 OLDEST 10 {pair_type.upper()} PAIRS:")
        for i, pair in enumerate(pairs_with_dates[:10], 1):
            print(f"{i:2d}. {pair['symbol']:15s} - {pair['listing_date']}")
        
        print(f"\n🆕 NEWEST 10 {pair_type.upper()} PAIRS:")
        for i, pair in enumerate(pairs_with_dates[-10:], len(pairs_with_dates)-9):
            print(f"{i:2d}. {pair['symbol']:15s} - {pair['listing_date']}")

def main():
    """Main function"""
    print("=" * 80)
    print("🎯 BINANCE COIN LISTING ANALYZER WITH DATES")
    print("🔍 Fetching all pairs with their listing dates")
    print("=" * 80)
    
    # # Get spot pairs
    # print("\n📈 PROCESSING SPOT PAIRS")
    # print("-" * 40)
    # spot_pairs = get_spot_pairs_with_dates()
    # spot_pairs = fetch_listing_dates(spot_pairs, is_futures=False)
    # spot_pairs = sort_pairs_by_date(spot_pairs)
    
    # Get futures pairs
    print("\n🚀 PROCESSING FUTURES PAIRS")
    print("-" * 40)
    futures_pairs = get_futures_pairs_with_dates()
    print(f"🎯 TOTAL COINS TO PROCESS: {len(futures_pairs)}")
    print(f"⏱️  Estimated time: {len(futures_pairs) * 0.02 / 60:.1f} minutes")
    print("-" * 40)
    futures_pairs = fetch_listing_dates(futures_pairs, is_futures=True)
    futures_pairs = sort_pairs_by_date(futures_pairs)
    
    # Save results
    print("\n💾 SAVING RESULTS")
    print("-" * 40)
    
    # Save to JSON
    # save_to_json(spot_pairs, 'spot_with_dates.json')
    save_to_json(futures_pairs, 'futures_with_dates.json')
    
    # Save to CSV for easy viewing
    # save_to_csv(spot_pairs, 'spot_with_dates.csv')
    save_to_csv(futures_pairs, 'futures_with_dates.csv')
    
    # Print summaries
    # print_summary(spot_pairs, "spot")
    print_summary(futures_pairs, "futures")
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Files saved:")
    print(f"   - spot_with_dates.json & .csv")
    print(f"   - futures_with_dates.json & .csv")

if __name__ == "__main__":
    main() 