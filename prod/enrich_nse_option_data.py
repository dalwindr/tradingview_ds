"""
oc_advanced_analysis.py
Advanced option-chain analysis:
 - Fetches CSV from GitHub raw URL (or local path)
 - Prepares input, runs multi-factor scoring
 - Produces explicit trade signals: BUY CALL @ strike / BUY PUT @ strike
 - Saves dashboard PNG and CSV with enriched per-strike metrics
"""

import io
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# -------------------------
# INPUT PREPARATION LAYER
# -------------------------
def load_csv_from_github(raw_url_or_path):
    """
    Accepts either:
      - a raw GitHub URL (https://raw.githubusercontent.com/.../file.csv)
      - or a local file path
    Returns: pandas.DataFrame
    """
    if str(raw_url_or_path).lower().startswith("http"):
        r = requests.get(raw_url_or_path, timeout=30)
        r.raise_for_status()
        content = io.StringIO(r.text)
        df = pd.read_csv(content)
    else:
        df = pd.read_csv(raw_url_or_path)
    return df

def normalize_columns(df):
    """
    Make a best-effort mapping from common export column names -> canonical names.
    Required canonical columns after this:
      strike, call_oi, put_oi, call_oi_chg, put_oi_chg, call_ltp, put_ltp, call_iv, put_iv, expiry
    """
    colmap = {
        'CE OI':'call_oi','PE OI':'put_oi',
        'CE Chng OI':'call_oi_chg','PE Chng OI':'put_oi_chg',
        'CE LTP':'call_ltp','PE LTP':'put_ltp',
        'CE IV':'call_iv','PE IV':'put_iv',
        'strike_price':'strike','Strike':'strike','STRIKE':'strike',
        'expiry_dt':'expiry','expiry_date':'expiry'
    }
    df = df.rename(columns={c:colmap[c] for c in df.columns if c in colmap})
    # Ensure numeric
    for c in ['strike','call_oi','put_oi','call_oi_chg','put_oi_chg','call_ltp','put_ltp','call_iv','put_iv']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else:
            df[c] = 0
    # If expiry not available, try to infer or set NaT
    if 'expiry' in df.columns:
        try:
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
        except:
            df['expiry'] = pd.NaT
    else:
        df['expiry'] = pd.NaT
    return df

# -------------------------
# CORE PROCESSING LOGIC
# -------------------------
def compute_scores(df, spot_price, today=None):
    """
    Adds these columns to df:
      - oi_diff
      - distance (abs strike - spot)
      - support_score, resistance_score (composite)
      - iv_skew = put_iv - call_iv
      - tte_days (time to expiry)
    Returns enriched df and summary dict.
    """
    df = df.copy()
    df['oi_diff'] = df['call_oi'] - df['put_oi']
    df['distance'] = (df['strike'] - spot_price).abs()
    # time to expiry
    if today is None:
        today = pd.Timestamp.now().normalize()
    df['tte_days'] = (df['expiry'] - today).dt.days.fillna(0).clip(lower=0)
    # small smoothing to avoid divide-by-zero
    df['support_score'] = df['put_oi'] * (1 + df['put_oi_chg'] / (abs(df['put_oi']) + 1))
    df['resistance_score'] = df['call_oi'] * (1 + df['call_oi_chg'] / (abs(df['call_oi']) + 1))
    # reward strikes close to spot slightly
    df['proximity_bonus'] = 1 / (1 + df['distance']/50.0)  # tuneable
    df['support_score'] *= df['proximity_bonus']
    df['resistance_score'] *= df['proximity_bonus']
    # IV skew
    df['iv_skew'] = df['put_iv'] - df['call_iv']
    # liquidity filter
    df['total_oi'] = df['call_oi'] + df['put_oi']
    # normalized ranks for scoring
    df['support_rank'] = df['support_score'].rank(method='min', ascending=False)
    df['resistance_rank'] = df['resistance_score'].rank(method='min', ascending=False)
    # Output summary metrics
    summary = {}
    summary['total_call_oi'] = int(df['call_oi'].sum())
    summary['total_put_oi'] = int(df['put_oi'].sum())
    summary['pcr'] = summary['total_put_oi'] / summary['total_call_oi'] if summary['total_call_oi']>0 else float('nan')
    summary['avg_call_iv'] = float(df['call_iv'].replace(0,np.nan).mean())
    summary['avg_put_iv'] = float(df['put_iv'].replace(0,np.nan).mean())
    # Top supports / resistances
    summary['top_supports'] = df[df['strike'] <= spot_price].nlargest(5, 'support_score')[['strike','support_score','put_oi','put_oi_chg','put_iv','tte_days']].reset_index(drop=True)
    summary['top_resistances'] = df[df['strike'] >= spot_price].nlargest(5, 'resistance_score')[['strike','resistance_score','call_oi','call_oi_chg','call_iv','tte_days']].reset_index(drop=True)
    # ATM
    summary['atm_strike'] = int(df.iloc[df['distance'].idxmin()]['strike'])
    summary['enriched_df'] = df
    return df, summary

def determine_final_signals(summary, df, spot_price, config=None):
    """
    Multi-factor rule to determine clear trade signals.
    Returns: list of candidate signals (each a dict with type, strike, reason, confidence)
    Rule outline (configurable):
      - Confirm at least 2 of: PCR threshold, fresh OI change in support/resistance, IV skew sign
      - Prefer strikes with high support/resistance score AND sufficient liquidity (total_oi)
      - TTE filter: ignore strikes with TTE > X days for intraday signals (configurable)
    """
    if config is None:
        config = {
            'pcr_bull_thresh': 1.1,   # put oi dominates -> bullish
            'pcr_bear_thresh': 0.9,   # call oi dominates -> bearish
            'min_total_oi': 5000,     # strike liquidity threshold
            'max_tte_days': 10,       # consider near expiries primarily
            'min_confidence': 1       # min rules matched
        }

    signals = []
    pcr = summary['pcr']
    top_supports = summary['top_supports']
    top_resistances = summary['top_resistances']

    # Check market-wide drivers
    put_chg_sum = df['put_oi_chg'].sum()
    call_chg_sum = df['call_oi_chg'].sum()

    # Candidate LONG CALL signals (buy calls from strong support)
    if pcr >= config['pcr_bull_thresh'] or put_chg_sum > call_chg_sum:
        # iterate top supports, pick the highest scoring that meets liquidity and tte
        for _, row in top_supports.iterrows():
            if row['put_oi'] + 0 >= config['min_total_oi']:
                if row['tte_days'] <= config['max_tte_days'] or row['tte_days']==0:
                    # confidence scoring
                    confidence = 0
                    if pcr >= config['pcr_bull_thresh']: confidence += 1
                    if row['put_oi_chg'] > 0: confidence += 1
                    if row['put_iv'] >= summary['avg_put_iv']: confidence += 1
                    if confidence >= config['min_confidence']:
                        signals.append({
                            'side':'LONG_CALL',
                            'strike':int(row['strike']),
                            'confidence':confidence,
                            'reason':f"Support score {row['support_score']:.0f}, put_oi {int(row['put_oi'])}, put_oi_chg {row['put_oi_chg']:.1f}, pcr {pcr:.2f}"
                        })
                        break

    # Candidate LONG PUT signals (buy puts from strong resistance)
    if pcr <= config['pcr_bear_thresh'] or call_chg_sum > put_chg_sum:
        for _, row in top_resistances.iterrows():
            if row['call_oi'] + 0 >= config['min_total_oi']:
                if row['tte_days'] <= config['max_tte_days'] or row['tte_days']==0:
                    confidence = 0
                    if pcr <= config['pcr_bear_thresh']: confidence += 1
                    if row['call_oi_chg'] > 0: confidence += 1
                    if row['call_iv'] >= summary['avg_call_iv']: confidence += 1
                    if confidence >= config['min_confidence']:
                        signals.append({
                            'side':'LONG_PUT',
                            'strike':int(row['strike']),
                            'confidence':confidence,
                            'reason':f"Resistance score {row['resistance_score']:.0f}, call_oi {int(row['call_oi'])}, call_oi_chg {row['call_oi_chg']:.1f}, pcr {pcr:.2f}"
                        })
                        break

    # If nothing found, add neutral suggestion (straddle near ATM) with reason
    if len(signals)==0:
        signals.append({
            'side':'NEUTRAL',
            'strike': summary['atm_strike'],
            'confidence':0,
            'reason':'No strong confluence — consider straddle/avoid directional trade'
        })

    return signals

# -------------------------
# OUTPUT PREPARATION LAYER
# -------------------------
def make_dashboard_and_save(enriched_df, summary, signals, asset_name='ASSET', out_png='oc_advanced_dashboard.png'):
    """Creates a dashboard PNG with OI chart and report box + final signals"""
    atm_strike = summary['atm_strike']
    spot = enriched_df['strike'].iloc[0] if False else None  # not used
    plt.figure(figsize=(14,9))
    gs = GridSpec(2,2, width_ratios=[2,1])
    ax0 = plt.subplot(gs[:,:1])
    # sort df by strike
    plot_df = enriched_df.sort_values('strike')
    # Plot call vs put OI as mirrored bars for Groww-like appearance
    strikes = plot_df['strike'].astype(int)
    calls = plot_df['call_oi']
    puts = plot_df['put_oi']
    # Horizontal plotting
    ax0.barh(strikes, calls, align='center', label='Call OI', color='tab:red', alpha=0.8)
    ax0.barh(strikes, -puts, align='center', label='Put OI', color='tab:green', alpha=0.8)
    ax0.axhline(atm_strike, color='blue', linestyle='--', label=f'ATM {atm_strike}')
    ax0.set_xlabel('Open Interest (calls positive, puts negative)')
    ax0.set_ylabel('Strike')
    ax0.set_title(f'{asset_name} Option Chain OI')
    ax0.legend()

    # report box
    ax1 = plt.subplot(gs[0,1]); ax1.axis('off')
    txt = (
        f"Spot: provided\nATM: {summary['atm_strike']}\n"
        f"Total Call OI: {summary['total_call_oi']}\n"
        f"Total Put OI: {summary['total_put_oi']}\n"
        f"PCR: {summary['pcr']:.3f}\n"
        f"Avg Call IV: {summary['avg_call_iv']:.2f}\n"
        f"Avg Put IV: {summary['avg_put_iv']:.2f}\n"
    )
    ax1.text(0,1,txt, fontsize=10, va='top')

    # signal box
    ax2 = plt.subplot(gs[1,1]); ax2.axis('off')
    stext = "FINAL SIGNALS:\n\n"
    for s in signals:
        stext += f"{s['side']} @ {s['strike']}  (conf {s['confidence']})\n  - {s['reason']}\n\n"
    ax2.text(0,1, stext, fontsize=10, va='top')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png

# -------------------------
# RUN WRAPPER
# -------------------------
def run_from_github(raw_csv_url, spot_price, asset_name='ASSET', out_png='oc_advanced_dashboard.png'):
    df = load_csv_from_github(raw_csv_url)
    df = normalize_columns(df)
    enriched_df, summary = compute_scores(df, spot_price)
    signals = determine_final_signals(summary, enriched_df, spot_price)
    dashboard = make_dashboard_and_save(enriched_df, summary, signals, asset_name, out_png)
    # save enriched csv for records
    enriched_out = out_png.replace('.png','.enriched.csv')
    enriched_df.to_csv(enriched_out, index=False)
    result = {
        'summary': summary,
        'signals': signals,
        'dashboard': dashboard,
        'enriched_csv': enriched_out
    }
    return result

# -------------------------
# Example usage (replace with your real URL & spot)
# -------------------------
if __name__ == '__main__':
    # EXAMPLE: raw GitHub csv URL -> replace with the repo you trust
    GITHUB_RAW_CSV_URL = 'https://raw.githubusercontent.com/<user>/<repo>/main/options_snapshot.csv'
    SPOT_PRICE = 81548.73
    # Uncomment to run:
    # out = run_from_github(GITHUB_RAW_CSV_URL, SPOT_PRICE, asset_name='NIFTY')
    # print(out['signals'])
