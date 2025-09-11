import requests
import pandas as pd

def fetch_nse_oc(symbol="NIFTY", expiry=None):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    sess = requests.Session()
    data = sess.get(url, headers=headers).json()

    # Parse into rows
    records = []
    for d in data['records']['data']:
        strike = d['strikePrice']
        expiry_dt = d['expiryDate']
        ce = d.get('CE', {})
        pe = d.get('PE', {})
        rec = {
            'strike': strike,
            'expiry': expiry_dt,
            'call_oi': ce.get('openInterest', 0),
            'call_oi_chg': ce.get('changeinOpenInterest', 0),
            'call_ltp': ce.get('lastPrice', 0),
            'call_iv': ce.get('impliedVolatility', 0),
            'put_oi': pe.get('openInterest', 0),
            'put_oi_chg': pe.get('changeinOpenInterest', 0),
            'put_ltp': pe.get('lastPrice', 0),
            'put_iv': pe.get('impliedVolatility', 0),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    # Convert expiry to datetime
    df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce', dayfirst=True)
    return df
