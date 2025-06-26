import os, time, requests, datetime as dt, pandas as pd


# --- helpers from previous snippet ------------------------------------------

def to_millis(ts: str | int) -> int:
    if isinstance(ts, (int, float)):
        return int(ts)
    return int(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)


def get_binance_1s_klines(symbol: str,
                          start_ms: int,
                          end_ms:   int | None = None,
                          limit:    int = 1_000) -> list[list]:
    """Return raw kline rows (no parsing) between start_ms and end_ms."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol":   symbol.upper(),
        "interval": "1s",
        "limit":    limit,
        "startTime": start_ms,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    out = []
    while True:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        rows = res.json()
        if not rows:
            break
        out.extend(rows)
        params["startTime"] = rows[-1][6] + 1          # next millisecond
        if len(rows) < limit or (end_ms and params["startTime"] > end_ms):
            break
        time.sleep(0.25)                               # stay under rate-limit
    return out


# --- NEW convenience wrapper -------------------------------------------------

def dump_all_prices(symbol: str,
                    price_col: str = "close",          # 'open' | 'high' | 'low' | 'close'
                    out_path: str | None = None) -> str:
    """
    Fetch **all** historical 1-second klines for `symbol`
    and output just the chosen price column (no dates) as CSV/TSV/plain-txt.
    Returns the absolute file-path.
    """
    # Binance returns an error if startTime precedes the listing date,
    # so we first ask for a single candle to discover the earliest timestamp.
    epoch_ms = 0
    first = get_binance_1s_klines(symbol, epoch_ms, limit=1)
    if not first:
        raise ValueError(f"No data for {symbol}")
    first_ts = first[0][0]

    # Pull everything up to 'now'
    now_ms = int(time.time() * 1000)
    raws = get_binance_1s_klines(symbol, first_ts, now_ms)

    # Map kline index → column name
    idx = {"open": 1, "high": 2, "low": 3, "close": 4}[price_col]
    prices = [float(row[idx]) for row in raws]

    if out_path is None:
        out_path = f"{symbol.upper()}_{price_col}_1s_full.txt"
    out_path = os.path.abspath(out_path)

    # one number per line, no header
    with open(out_path, "w") as f:
        f.write("\n".join(map(str, prices)))

    print(f"Wrote {len(prices):,} {price_col} prices to {out_path}")
    return out_path


# --- usage example -----------------------------------------------------------
if __name__ == "__main__":
    dump_all_prices(
        symbol="SAHARAUSDT",       # make sure it exists on Spot
        price_col="close",      # any of: open / high / low / close
        out_path="btc_close_1s_history.txt",
    )
