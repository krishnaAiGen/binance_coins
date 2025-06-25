import requests
import json

def get_spot_pairs():
    """Get all active spot trading pairs"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    pairs = []
    for symbol in data['symbols']:
        if symbol['status'] == 'TRADING':
            pairs.append(symbol['symbol'])
    
    return pairs

def get_futures_pairs():
    """Get all active futures trading pairs"""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    pairs = []
    for symbol in data['symbols']:
        if symbol['status'] == 'TRADING':
            pairs.append(symbol['symbol'])
    
    return pairs

# Get trading pairs
spot_pairs = get_spot_pairs()
futures_pairs = get_futures_pairs()

# Save to JSON files
with open('spot.json', 'w') as f:
    json.dump(spot_pairs, f, indent=2)

with open('futures.json', 'w') as f:
    json.dump(futures_pairs, f, indent=2)

print(f"Spot pairs: {len(spot_pairs)}")
print(f"Futures pairs: {len(futures_pairs)}")
print("Files saved: spot.json, futures.json")